-- pandoc Lua filter: IEEE conference layout for the Word copy of the report.
-- Used by scripts/build_report.py together with the generated reference.docx.
--   * title and author block single-column, the rest in two columns;
--   * ::: wide blocks span both columns (continuous section breaks around them);
--   * roman-numbered sections (I., II.) and lettered subsections (A., B.);
--   * numbered equations, hanging-indent references, the abstract in the body.

local ROMAN = { "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
  "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX" }
local COLUMN_IN, PAGE_IN = 3.45, 7.0 -- usable width of one column and of the page, inches

local function section_break(cols)
  return pandoc.RawBlock("openxml", string.format(
    '<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" w:lineRule="exact"/>'
    .. '<w:rPr><w:sz w:val="2"/></w:rPr><w:sectPr><w:type w:val="continuous"/>'
    .. '<w:pgSz w:w="11906" w:h="16838"/>'
    .. '<w:pgMar w:top="1077" w:right="812" w:bottom="1361" w:left="812" w:header="567" w:footer="680" w:gutter="0"/>'
    .. '<w:cols w:num="%d" w:space="239"/></w:sectPr></w:pPr></w:p>', cols))
end

local function absolute_width(base)
  return {
    Image = function(img)
      local w = img.attributes.width
      if w and w:match("%%$") then
        img.attributes.width = string.format("%.2fin", tonumber(w:sub(1, -2)) / 100 * base)
      end
      return img
    end,
  }
end

local function esc(s)
  return (s:gsub("&", "&amp;"):gsub("<", "&lt;"):gsub(">", "&gt;"))
end

local function author_table(names, affiliation, city)
  local cell = function(name)
    local p = function(text, size, italic)
      return '<w:p><w:pPr><w:pStyle w:val="AuthorBlock"/></w:pPr><w:r><w:rPr>'
        .. (italic and "<w:i/>" or "") .. '<w:sz w:val="' .. size .. '"/></w:rPr><w:t>'
        .. esc(text) .. "</w:t></w:r></w:p>"
    end
    return '<w:tc><w:tcPr><w:tcW w:w="1250" w:type="pct"/></w:tcPr>'
      .. p(name, 22, false) .. p(affiliation, 18, true) .. p(city, 18, true) .. "</w:tc>"
  end
  local cells, grid = {}, {}
  for _, n in ipairs(names) do
    cells[#cells + 1] = cell(n)
    grid[#grid + 1] = '<w:gridCol w:w="' .. math.floor(10282 / #names) .. '"/>'
  end
  local none = '<w:top w:val="nil"/><w:left w:val="nil"/><w:bottom w:val="nil"/><w:right w:val="nil"/>'
    .. '<w:insideH w:val="nil"/><w:insideV w:val="nil"/>'
  return pandoc.RawBlock("openxml", '<w:tbl><w:tblPr><w:tblW w:w="5000" w:type="pct"/><w:jc w:val="center"/>'
    .. "<w:tblBorders>" .. none .. '</w:tblBorders><w:tblLayout w:type="fixed"/>'
    .. '<w:tblLook w:firstRow="0" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="1" w:noVBand="1"/>'
    .. "</w:tblPr><w:tblGrid>"
    .. table.concat(grid) .. "</w:tblGrid><w:tr>" .. table.concat(cells) .. "</w:tr></w:tbl>")
end

-- Pass 1: wide blocks get page-width images and column breaks around them.
local wide = {
  Div = function(d)
    if not d.classes:includes("wide") then return nil end
    local body = d.content:walk(absolute_width(PAGE_IN))
    local out = pandoc.Blocks({ section_break(2) })
    out:extend(body)
    out:insert(section_break(1))
    return out
  end,
}

-- Pass 2: everything else.
local sec, sub, eqn = 0, 0, 0
local main = {
  traverse = "topdown",
  Header = function(h)
    if h.classes:includes("unnumbered") then return h end
    if h.level == 1 then
      sec, sub = sec + 1, 0
      h.content:insert(1, pandoc.Str(ROMAN[sec] .. ". "))
    elseif h.level == 2 then
      sub = sub + 1
      h.content:insert(1, pandoc.Str(string.char(64 + sub) .. ". "))
    end
    return h
  end,
  Div = function(d)
    if d.classes:includes("eq") then
      eqn = eqn + 1
      local tex
      d:walk({ Math = function(m) tex = m.text end })
      local tab = pandoc.RawInline("openxml", "<w:r><w:tab/></w:r>")
      -- two formulas joined by \quad do not fit one column: one line each
      local lines, paras = {}, {}
      for part in (tex .. "\\quad"):gmatch("(.-)\\quad") do lines[#lines + 1] = part end
      for i, part in ipairs(lines) do
        local inl = { tab, pandoc.Math("InlineMath", "\\displaystyle " .. part) }
        if i == #lines then
          inl[#inl + 1] = tab
          inl[#inl + 1] = pandoc.Str("(" .. eqn .. ")")
        end
        paras[#paras + 1] = pandoc.Para(inl)
      end
      return pandoc.Div(paras, { ["custom-style"] = "Equation" }), false
    elseif d.classes:includes("references") then
      return pandoc.Div(d.content, { ["custom-style"] = "Reference" }), false
    elseif d.classes:includes("algo") then
      return pandoc.Div(d.content, { ["custom-style"] = "Algorithm" })
    elseif d.classes:includes("lefttable") then
      return d.content
    end
  end,
  Image = absolute_width(COLUMN_IN).Image,
  -- let Word fit the columns to their content instead of the source dash widths
  Table = function(t)
    for i, spec in ipairs(t.colspecs) do t.colspecs[i] = { spec[1], pandoc.ColWidthDefault } end
    return t
  end,
  Pandoc = function(doc)
    local m = doc.meta
    local names = {}
    for _, a in ipairs(m.author or {}) do names[#names + 1] = pandoc.utils.stringify(a) end
    local front = pandoc.Blocks({ author_table(names, pandoc.utils.stringify(m.affiliation or ""),
      pandoc.utils.stringify(m.city or "")), section_break(1) })
    if m.abstract then
      local inl = pandoc.Inlines({ pandoc.Emph({ pandoc.Str("Abstract") }), pandoc.Str("—") })
      inl:extend(pandoc.utils.blocks_to_inlines(m.abstract))
      front:insert(pandoc.Div({ pandoc.Para(inl) }, { ["custom-style"] = "Abstract" }))
    end
    if m.keywords then
      local kw = {}
      for _, k in ipairs(m.keywords) do kw[#kw + 1] = pandoc.utils.stringify(k) end
      front:insert(pandoc.Div({ pandoc.Para({ pandoc.Emph({ pandoc.Str("Index Terms") }),
        pandoc.Str("—" .. table.concat(kw, ", ") .. ".") }) }, { ["custom-style"] = "Abstract" }))
    end
    front:extend(doc.blocks)
    doc.blocks = front
    -- author stays in the metadata for the file properties; build_report.py drops
    -- the plain author paragraphs pandoc writes under the title
    m.abstract, m.date, m.subtitle = nil, nil, nil
    return doc
  end,
}

return { wide, main }
