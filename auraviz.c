-- AuraViz Control Panel v14.0
-- GL-only edition — no engine switching needed
-- Place in: C:\Program Files\VideoLAN\VLC\lua\extensions\

dlg = nil
preset_label = nil
preset_dropdown = nil
ontop_checkbox = nil
chosen_preset = -1
always_on_top = true

-- Persistence ----------------------------------------------------------------
function settings_path()
    local sep = package.config:sub(1,1) or "\\"
    return vlc.config.userdatadir() .. sep .. "auraviz_settings.txt"
end

function save_settings()
    local ok, err = pcall(function()
        local f = io.open(settings_path(), "w")
        if f then
            f:write(chosen_preset .. "\n" .. (always_on_top and "1" or "0") .. "\n")
            f:close()
        end
    end)
    if not ok then vlc.msg.warn("AuraViz: save_settings failed: " .. tostring(err)) end
end

function load_settings()
    local ok, err = pcall(function()
        local f = io.open(settings_path(), "r")
        if f then
            chosen_preset = tonumber(f:read("*l")) or -1
            always_on_top = (f:read("*l") or "1") == "1"
            f:close()
        end
    end)
    if not ok then vlc.msg.warn("AuraViz: load_settings failed: " .. tostring(err)) end
    if not is_valid_preset(chosen_preset) then chosen_preset = -1 end
end

function push_config()
    local ok, err = pcall(function()
        vlc.config.set("audio-visual", "auraviz")
        vlc.config.set("auraviz-preset", chosen_preset)
        vlc.config.set("auraviz-ontop", always_on_top)
    end)
    if not ok then vlc.msg.warn("AuraViz: push_config failed: " .. tostring(err)) end
    save_settings()
end

-- Open URL -------------------------------------------------------------------
function open_url(url)
    local sep = package.config:sub(1,1) or "\\"
    if sep == "\\" then os.execute('start "" "' .. url .. '"')
    else os.execute('xdg-open "' .. url .. '" &') end
end

-- Presets (grouped by theme, matching C switch order) ------------------------
local P = {
    {-1, "Auto Cycle (Shuffle)"},
    {44, "Fractal Vortex Inferno"},
    {45, "Lava Lightning"},
    {46, "Disco Inferno"},
    {47, "Inferno Bubbles"},
    {48, "Helix Inferno"},
    {49, "Fire Constellation"},
    {50, "Lava Maze"},
    {51, "Nebula Lightning Storm"},
    {52, "Cosmic Jellyfish"},
    {53, "Nebula Forge"},
    {54, "Nebula Windows"},
    {55, "Galactic Smoke"},
    {56, "Jellyfish Matrix"},
    {57, "Kaleidoscope Nebula"},
    {58, "Crystal Galaxy"},
    {59, "Solar Flare"},
    {60, "Solar Wind"},
    {61, "Quantum Tunnel"},
    {62, "Wormhole"},
    {63, "Bubble Tunnel"},
    {64, "Matrix Tunnel"},
    {65, "Warp Drive"},
    {66, "Plasma Vortex Tunnel"},
    {67, "Stargate"},
    {68, "Void Reactor"},
    {69, "Vortex Starfield"},
    {70, "Thunder Dome"},
    {71, "Supercell"},
    {72, "Neon Asteroid Storm"},
    {73, "Polyhedra Storm"},
    {74, "Neon Pulse Grid"},
    {75, "Molten Kaleidoscope"},
    {76, "Smoke & Mirrors"},
    {77, "Prism Cascade"},
    {78, "Glitch Aurora"},
    {79, "Fractal Constellation"},
    {80, "Fractal Matrix"},
    {81, "Fractal Ocean"},
    {82, "Spiral Fractal Warp"},
    {83, "Bioluminescent Reef"},
    {84, "Toxic Swamp"},
    {85, "Galactic DNA"},
    {86, "Plasma Web"},
    {87, "Phantom Grid"},
    {88, "Retro Wave"},
    {89, "Ghost Ship"},
    {90, "Neon Ripple Maze"},
    {91, "Helix Supernova"},
    {92, "Electric Helix"},
    {93, "Ember Drift"},
    {94, "Acid Rain"},
    {95, "Comet Shower"},
    {96, "Neon Jungle"},
    {97, "Pulsar"},
    {98, "Lava Bubble"},
    {99, "Quantum Field"},
    {100, "Blue Inferno Tunnel"},
    {101, "Galaxy Kaleidoscope"},
    {102, "Firefly Swamp"},
    {103, "Plasma Rings"},
    {104, "Glitch Maze"},
    {105, "Starfield Warp"},
    {106, "Fireball Galaxy"},
    {107, "Fractal Storm"},
    {108, "Retro Arcade"},
    {109, "Smoke Nebula"},
    {110, "Ripple Storm"},
    {111, "Helix Tunnel"},
    {112, "Star Grid"},
    {113, "Volcanic Vortex"},
    {114, "Polyhedra Plasma"},
    {115, "Matrix Aurora"},
    {116, "Shockwave Kaleidoscope"},
    {117, "Spectrum Vortex"},
    {118, "Bubble Kaleidoscope"},
    {119, "Stardust Helix"},
    {120, "Smoke Ripple"},
    {121, "Neon Circuit"},
    {122, "Aurora Tunnel"},
    {123, "Green Lightning Storm"},
    {124, "Plasma Helix"},
    {125, "Firework Tunnel"},
    {126, "Glitch Plasma"},
    {127, "Win98 Storm"},
    {128, "Cosmic Ripple"},
    {129, "Lava Constellation"},
    {130, "Julia Vortex"},
    {131, "Asteroid Kaleidoscope"},
    {132, "Particle Wave"},
    {133, "Polyhedra Fire"},
    {134, "Bubble Matrix"},
    {135, "Constellation Storm"},
    {136, "Inferno Ripple"},
    {137, "Neon Supernova"},
}

function pname(id)
    for _, p in ipairs(P) do if p[1] == id then return p[2] end end
    return "Unknown"
end

function preset_index()
    for i, p in ipairs(P) do if p[1] == chosen_preset then return i end end
    return 1
end

function is_valid_preset(id)
    for _, p in ipairs(P) do if p[1] == id then return true end end
    return false
end

-- HTML UI --------------------------------------------------------------------
function build_ui()
    local ok_ui, vis = pcall(function() return vlc.config.get("audio-visual") or "" end)
    if not ok_ui then vis = "" end
    local on = (vis == "auraviz")
    local sc = on and "#00e676" or "#e94560"
    local dot = on and "&#9679;" or "&#9675;"
    local stxt = on and "ACTIVE" or "OFF"

    local h = '<html><head><style>body{background-color:#0d0d1a;margin:0;padding:0;}</style></head><body>'
    h = h .. '<table width="100%" cellpadding="0" cellspacing="0">'

    -- Header
    h = h .. '<tr><td style="background-color:#111128;padding:8px 0 5px 0;" align="center">'
    h = h .. '<span style="font-size:16px;font-weight:bold;">'
    h = h .. '<span style="color:#00e5ff;">&#9733; AURA</span>'
    h = h .. '<span style="color:#e94560;">VIZ &#9733;</span></span>'
    h = h .. '</td></tr>'

    -- Status + Engine badge
    h = h .. '<tr><td style="background-color:#151530;padding:5px 10px;" align="center">'
    h = h .. '<span style="color:' .. sc .. ';font-size:12px;font-weight:bold;">' .. dot .. ' ' .. stxt .. '</span>'
    h = h .. '&nbsp;&nbsp;<span style="color:#b388ff;font-size:11px;font-weight:bold;">OpenGL</span>'
    h = h .. '</td></tr>'

    -- Preset header
    h = h .. '<tr><td style="background-color:#0d0d1a;padding:5px 8px 3px 8px;">'
    h = h .. '<span style="color:#e94560;font-size:11px;font-weight:bold;">&#9656; PRESET</span>'
    h = h .. '&nbsp;<span style="color:#444466;font-size:9px;">select from list or use arrows</span>'
    h = h .. '</td></tr>'

    -- Coffee button
    h = h .. '<tr><td style="background-color:#111128;padding:6px 8px;" align="center">'
    h = h .. '<table border="1" cellpadding="4" cellspacing="0" style="border-color:#ffab40;border-style:solid;border-collapse:collapse;"><tr><td align="center">'
    h = h .. '<a href="https://buymeacoffee.com/davekaiser" style="color:#ffab40;font-size:12px;text-decoration:none;">&#9749; Buy me a Coffee</a>'
    h = h .. '</td></tr></table>'
    h = h .. '</td></tr>'

    -- First run caution
    h = h .. '<tr><td style="background-color:#0d0d1a;padding:6px 8px;" align="center">'
    h = h .. '<span style="color:#ffd600;font-size:11px;">&#9888;</span> '
    h = h .. '<span style="color:#ff1744;font-size:11px;font-weight:bold;">First run: You must close and reopen VLC to enable the extension</span>'
    h = h .. '</td></tr>'

    h = h .. '</table></body></html>'
    return h
end

-- Dialog ---------------------------------------------------------------------
function show_dialog()
    if dlg then dlg:delete(); dlg = nil end
    dlg = vlc.dialog("AuraViz")
    dlg:add_html(build_ui(), 1, 1, 5, 1)

    -- Prev button
    dlg:add_button("<<", click_prev, 1, 2, 1, 1)

    -- Current preset label
    preset_label = dlg:add_label("<center><b>" .. pname(chosen_preset) .. "</b></center>", 2, 2, 3, 1)

    -- >> button
    dlg:add_button(">>", click_next, 5, 2, 1, 1)

    -- Jump-to dropdown + Go button
    preset_dropdown = dlg:add_dropdown(1, 3, 4, 1)
    for i, p in ipairs(P) do
        preset_dropdown:add_value(p[2], i)
    end
    dlg:add_button("Go", click_go, 5, 3, 1, 1)

    -- Always on top checkbox
    ontop_checkbox = dlg:add_check_box("AuraViz window always on top", always_on_top, 1, 4, 5, 1)

    dlg:show()
end

-- Handlers -------------------------------------------------------------------
function click_prev()
    local idx = preset_index() - 1
    if idx < 1 then idx = #P end
    chosen_preset = P[idx][1]
    push_config()
    if preset_label then preset_label:set_text("<center><b>" .. pname(chosen_preset) .. "</b></center>") end
end

function click_next()
    local idx = preset_index() + 1
    if idx > #P then idx = 1 end
    chosen_preset = P[idx][1]
    push_config()
    if preset_label then preset_label:set_text("<center><b>" .. pname(chosen_preset) .. "</b></center>") end
end

function click_go()
    if not preset_dropdown then return end
    local ok, err = pcall(function()
        local sel = preset_dropdown:get_value()
        if sel and sel >= 1 and sel <= #P then
            chosen_preset = P[sel][1]
            push_config()
            if preset_label then preset_label:set_text("<center><b>" .. pname(chosen_preset) .. "</b></center>") end
        else
            vlc.msg.warn("AuraViz: invalid dropdown selection: " .. tostring(sel))
        end
    end)
    if not ok then vlc.msg.warn("AuraViz: click_go failed: " .. tostring(err)) end
end

function sync_ontop()
    if ontop_checkbox then
        local ok, err = pcall(function()
            always_on_top = ontop_checkbox:get_checked()
            vlc.config.set("auraviz-ontop", always_on_top)
            save_settings()
        end)
        if not ok then vlc.msg.warn("AuraViz: sync_ontop failed: " .. tostring(err)) end
    end
end

-- VLC Hooks ------------------------------------------------------------------
function descriptor()
    return {
        title = "AuraViz",
        version = "14.0",
        author = "AuraViz",
        capabilities = {"menu", "input-listener"},
        description = "AuraViz audio visualization control (OpenGL)"
    }
end

function activate()
    load_settings()
    push_config()
    show_dialog()
end

function deactivate()
    sync_ontop()
    save_settings()
    if dlg then dlg:delete(); dlg = nil end
end

function close()
    sync_ontop()
    push_config()
end

function menu() return {"Show Settings", "Buy me a Coffee"} end

function trigger_menu(id)
    load_settings()
    if id == 1 then show_dialog()
    elseif id == 2 then open_url("https://buymeacoffee.com/davekaiser") end
end

function input_changed()
    sync_ontop()
    load_settings()
    local ok, err = pcall(function()
        if vlc.config.get("audio-visual") == "auraviz" then
            vlc.config.set("auraviz-preset", chosen_preset)
            vlc.config.set("auraviz-ontop", always_on_top)
        end
    end)
    if not ok then vlc.msg.warn("AuraViz: input_changed failed: " .. tostring(err)) end
end
