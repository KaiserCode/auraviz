-- AuraViz Control Panel v8.0
-- Place in: C:\Program Files\VideoLAN\VLC\lua\extensions\

dlg = nil
dd = nil
render_mode = "cpu"
chosen_preset = 0

-- Persistence ----------------------------------------------------------------
function settings_path()
    local sep = package.config:sub(1,1) or "\\"
    return vlc.config.userdatadir() .. sep .. "auraviz_settings.txt"
end

function save_settings()
    local f = io.open(settings_path(), "w")
    if f then
        f:write(chosen_preset .. "\n" .. render_mode .. "\n")
        f:close()
    end
end

function load_settings()
    local f = io.open(settings_path(), "r")
    if f then
        local p = f:read("*l")
        local m = f:read("*l")
        f:close()
        chosen_preset = tonumber(p) or 0
        render_mode = m or "cpu"
    end
end

function push_config()
    local vis = render_mode == "opengl" and "auraviz_gl" or "auraviz"
    vlc.config.set("audio-visual", vis)
    vlc.config.set("auraviz-preset", chosen_preset)
    save_settings()
end

function grab_dropdown()
    if dd then
        local v = dd:get_value()
        if v then chosen_preset = v end
    end
end

-- Presets (plain ASCII, no emojis, no escapes) -------------------------------
local P = {
    {0,  "Auto Cycle"},
    {1,  "Spectrum Bars"},
    {2,  "Waveform"},
    {3,  "Circular"},
    {4,  "Particles"},
    {5,  "Nebula"},
    {6,  "Plasma"},
    {7,  "Tunnel"},
    {8,  "Kaleidoscope"},
    {9,  "Lava Lamp"},
    {10, "Starburst"},
    {11, "Electric Storm"},
    {12, "Ripple Pool"},
    {13, "Fractal Warp"},
    {14, "Spiral Galaxy"},
    {15, "Glitch Matrix"},
    {16, "Aurora Borealis"},
    {17, "Pulse Grid"},
    {18, "Fire"},
    {19, "Diamond Rain"},
    {20, "Vortex"},
}

function pname(id)
    for _,p in ipairs(P) do
        if p[1] == id then return p[2] end
    end
    return "Unknown"
end

-- HTML UI (all visual elements here) -----------------------------------------
function build_ui()
    local vis = vlc.config.get("audio-visual") or ""
    local on = (vis == "auraviz" or vis == "auraviz_gl")
    local sc = on and "#00e676" or "#e94560"
    local dot = on and "&#9679;" or "&#9675;"
    local stxt = on and "ACTIVE" or "OFF"

    local h = '<table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0d0d1a;">'

    -- Header
    h = h .. '<tr><td style="background-color:#111128;padding:8px 0 3px 0;" align="center">'
    h = h .. '<span style="font-size:18px;font-weight:bold;">'
    h = h .. '<span style="color:#00e5ff;">&#9733; AURA</span>'
    h = h .. '<span style="color:#e94560;">VIZ &#9733;</span></span>'
    h = h .. '</td></tr>'

    -- Status
    h = h .. '<tr><td style="background-color:#151530;padding:4px 12px;" align="center">'
    h = h .. '<span style="color:' .. sc .. ';font-size:11px;font-weight:bold;">' .. dot .. ' ' .. stxt .. '</span>'
    h = h .. '&nbsp;&nbsp;&nbsp;'
    h = h .. '<span style="color:#666688;font-size:10px;">Visual: </span>'
    h = h .. '<span style="color:#00e5ff;font-size:10px;font-weight:bold;">' .. pname(chosen_preset) .. '</span>'
    h = h .. '</td></tr>'

    -- Engine section header
    h = h .. '<tr><td style="background-color:#0d0d1a;padding:5px 12px 2px 12px;">'
    h = h .. '<span style="color:#e94560;font-size:9px;font-weight:bold;">&#9656; ENGINE</span>'
    h = h .. '</td></tr>'

    -- Engine toggle panels
    local cpu_on = (render_mode == "cpu")
    local gl_on = (render_mode == "opengl")

    local cpu_bg  = cpu_on and "#1a1a44" or "#0d0d1a"
    local cpu_brd = cpu_on and "#00e5ff" or "#333350"
    local cpu_c   = cpu_on and "#00e5ff" or "#555577"
    local cpu_d   = cpu_on and "&#9679; ON" or "&#9675; OFF"

    local gl_bg  = gl_on and "#1a1a44" or "#0d0d1a"
    local gl_brd = gl_on and "#b388ff" or "#333350"
    local gl_c   = gl_on and "#b388ff" or "#555577"
    local gl_d   = gl_on and "&#9679; ON" or "&#9675; OFF"

    h = h .. '<tr><td style="padding:0 8px 2px 8px;background-color:#0d0d1a;">'
    h = h .. '<table width="100%" cellpadding="0" cellspacing="3"><tr>'
    h = h .. '<td width="50%" style="background-color:' .. cpu_bg .. ';border:2px solid ' .. cpu_brd .. ';padding:6px 8px;">'
    h = h .. '<span style="color:' .. cpu_c .. ';font-size:12px;font-weight:bold;">CPU Pipeline</span><br/>'
    h = h .. '<span style="color:' .. cpu_c .. ';font-size:9px;">' .. cpu_d .. '</span>'
    h = h .. '</td>'
    h = h .. '<td width="50%" style="background-color:' .. gl_bg .. ';border:2px solid ' .. gl_brd .. ';padding:6px 8px;">'
    h = h .. '<span style="color:' .. gl_c .. ';font-size:12px;font-weight:bold;">OpenGL</span><br/>'
    h = h .. '<span style="color:' .. gl_c .. ';font-size:9px;">' .. gl_d .. '</span><br/>'
    h = h .. '<span style="color:#333350;font-size:7px;">coming soon</span>'
    h = h .. '</td>'
    h = h .. '</tr></table></td></tr>'

    -- Visual section header
    h = h .. '<tr><td style="background-color:#0d0d1a;padding:5px 12px 2px 12px;">'
    h = h .. '<span style="color:#e94560;font-size:9px;font-weight:bold;">&#9656; SELECT VISUAL</span>'
    h = h .. '&nbsp;&nbsp;<span style="color:#444466;font-size:8px;">(choose below, click Set)</span>'
    h = h .. '</td></tr>'

    -- Footer: coffee link
    h = h .. '<tr><td style="background-color:#111128;padding:5px 12px;" align="center">'
    h = h .. '<span style="color:#444466;font-size:8px;">Auto-saves &bull; Skip/restart track to apply</span>'
    h = h .. '<br/>'
    h = h .. '<a href="https://buymeacoffee.com/davekaiser" style="color:#ffab40;font-size:9px;text-decoration:none;">&#9749; Buy me a Coffee</a>'
    h = h .. '</td></tr>'

    h = h .. '</table>'
    return h
end

-- Dialog (minimal native widgets) --------------------------------------------
function show_dialog()
    if dlg then dlg:delete(); dlg = nil end
    dlg = vlc.dialog("AuraViz")

    -- Row 1: Full styled HTML panel
    dlg:add_html(build_ui(), 1, 1, 4, 1)

    -- Row 2: Dropdown + Set + Engine buttons
    dd = dlg:add_dropdown(1, 2, 3, 1)
    -- Current preset first so it shows as selected
    dd:add_value(pname(chosen_preset), chosen_preset)
    for _,p in ipairs(P) do
        if p[1] ~= chosen_preset then
            dd:add_value(p[2], p[1])
        end
    end

    -- Engine toggle button (shows what you'd switch TO)
    local eng_label = render_mode == "cpu" and "GPU" or "CPU"
    dlg:add_button(eng_label, click_engine_toggle, 4, 2, 1, 1)

    dlg:show()
end

-- Handlers -------------------------------------------------------------------
function click_set()
    grab_dropdown()
    push_config()
    show_dialog()
end

function click_engine_toggle()
    grab_dropdown()
    if render_mode == "cpu" then
        render_mode = "opengl"
    else
        render_mode = "cpu"
    end
    push_config()
    show_dialog()
end

-- VLC Hooks ------------------------------------------------------------------
function descriptor()
    return {
        title = "AuraViz",
        version = "8.0",
        author = "AuraViz",
        capabilities = {"menu", "input-listener"},
        description = "AuraViz audio visualization control"
    }
end

-- activate: always show dialog (whether first enable or re-enable)
function activate()
    load_settings()
    push_config()
    show_dialog()
end

function deactivate()
    if dd then grab_dropdown() end
    save_settings()
    if dlg then dlg:delete(); dlg = nil end
end

function close()
    if dd then grab_dropdown() end
    push_config()
end

-- Menu -----------------------------------------------------------------------
function menu()
    return {"Show Settings"}
end

function trigger_menu(id)
    load_settings()
    if id == 1 then
        show_dialog()
    end
end

function input_changed()
    load_settings()
    local vis = vlc.config.get("audio-visual") or ""
    if vis == "auraviz" or vis == "auraviz_gl" then
        vlc.config.set("auraviz-preset", chosen_preset)
    end
end
