function qfa_micro_report(csv_path, out_dir)
% qfa_micro_report - analysis + figures for the QFA micro-benchmark
% (plan: agent-workspace/randnla/project-plans/2026-09-02-qfa-micro-benchmark-plan.md,
% pass 10 FROZEN). Reads qfa_micro's CSV output, prints the a-priori
% reachability table FIRST, then the summary table, then renders exactly
% THREE PNGs (F1, F2, F3) to out_dir. Headless: no figure window is ever
% shown (DefaultFigureVisible is forced off below), so this is safe to run
% under `matlab -batch`.
%
% Usage: qfa_micro_report('qfa_calibrate.csv')
%        qfa_micro_report('qfa_calibrate.csv', 'figures/')
%
% DESIGN NOTE (per-trial pairing): every metric below that involves a
% fixed-arm "matvecs-to-tol" or a certified-arm "overshoot" is computed
% PER TRIAL FIRST (that trial's own error-vs-depth curve, that trial's own
% certified matvecs spend), matching the plan's "same trial, same probes"
% pairing requirement for metric 3 and the top-of-Metrics-section rule that
% every metric is "per (matrix, f, s, [tol], trial); aggregate = median over
% trials with 25/75." Only after that per-trial computation is a cell's
% number aggregated (median + 25/75) across the trials present.
%
% STATISTICS CAVEAT (stated once, applies everywhere below, per the plan):
% at n<=8 trials per cell, medians and 25/75 spreads are directional/
% qualitative, not precise CIs; win-counts are reported alongside medians
% where "how often" is the real question. Trial-to-trial spread reflects
% PROBE variance only -- matrix_seed fixes one Q rotation + spectrum draw
% per (spectrum,kappa) cell, so realization-to-realization variance in A
% itself is never sampled or represented in any spread/median/win-count here.

    if nargin < 1 || isempty(csv_path)
        error('qfa_micro_report:usage', 'csv_path is required');
    end
    if nargin < 2 || isempty(out_dir)
        out_dir = fullfile(fileparts(csv_path), 'figures');
    end
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    % Headless, always -- never a visible window (this is the one hard rule).
    set(0, 'DefaultFigureVisible', 'off');

    [T, n, viol] = load_csv(csv_path);

    matrices = {'geo1e3', 'geo1e6', 'logu1e6'};
    fnames   = {'sqrt', 'log1p'};
    svals    = [4 16];
    tols     = [1e-2 1e-4];

    fprintf('======================================================================\n');
    fprintf('QFA micro-benchmark report -- %s  (n=%d)\n', csv_path, n);
    fprintf('======================================================================\n\n');

    print_reachability_table(T, matrices, fnames, svals, tols, n);
    print_summary_table(T, viol, fnames, tols, n);

    make_F1(T, out_dir, n);
    make_F2(T, out_dir);
    make_F3(T, out_dir, viol);

    fprintf('\nqfa_micro_report: three figures written to %s\n', out_dir);
end

% =====================================================================
% [CSV loading -- skips '#' comment lines anywhere in the file (header
%  metadata at the top, metric7_violation trailer at the bottom)]
% =====================================================================
function [T, n, viol] = load_csv(csv_path)
    raw = fileread(csv_path);
    lines = regexp(raw, '\r\n|\n|\r', 'split');
    lines = lines(~cellfun(@isempty, lines));
    is_comment = startsWith(lines, '#');
    data_lines = lines(~is_comment);
    comment_lines = lines(is_comment);

    n = NaN;
    for i = 1:numel(comment_lines)
        tok = regexp(comment_lines{i}, '^#\s*n=(\d+)', 'tokens', 'once');
        if ~isempty(tok), n = str2double(tok{1}); break; end
    end
    if isnan(n)
        error('qfa_micro_report:load_csv', 'could not find "# n=" in header comments of %s', csv_path);
    end

    viol = struct('matrix', {}, 'f', {}, 's', {}, 'tol', {}, 'arm', {}, ...
                  'violations', {}, 'checked', {});
    for i = 1:numel(comment_lines)
        if startsWith(comment_lines{i}, '# metric7_violation ')
            m = regexp(comment_lines{i}, ...
                ['matrix=(?<matrix>\S+)\s+f=(?<f>\S+)\s+s=(?<s>\S+)\s+tol=(?<tol>\S+)\s+' ...
                 'arm=(?<arm>\S+)\s+violations=(?<violations>\d+)\s+checked=(?<checked>\d+)'], ...
                'names');
            if ~isempty(m)
                k = numel(viol) + 1;
                viol(k).matrix     = m.matrix;
                viol(k).f          = m.f;
                viol(k).s          = str2double(m.s);
                viol(k).tol        = str2double(m.tol);
                viol(k).arm        = m.arm;
                viol(k).violations = str2double(m.violations);
                viol(k).checked    = str2double(m.checked);
            end
        end
    end

    if isempty(data_lines)
        error('qfa_micro_report:load_csv', 'no data rows found in %s', csv_path);
    end
    tmpfile = [tempname() '.csv'];
    fid = fopen(tmpfile, 'w');
    fprintf(fid, '%s\n', data_lines{:});
    fclose(fid);
    T = readtable(tmpfile, 'Delimiter', ',', 'ReadVariableNames', true);
    delete(tmpfile);

    % Normalize string columns to cellstr regardless of readtable's guess
    % (string array vs cellstr depends on MATLAB version / column content).
    for col = {'matrix', 'f', 'arm'}
        c = col{1};
        if iscategorical(T.(c)) || isstring(T.(c))
            T.(c) = cellstr(T.(c));
        end
    end
end

% =====================================================================
% [Shared helpers]
% =====================================================================
function w = block_wall(n, s)
    w = floor(n / s);
end

% d*_interp(tol) from one trial's (depths, rel_err) curve. FIRST crossing
% scanning increasing d (pass 8 F8.2c, non-monotone near the roundoff floor).
% status: "ok" | "censored_low" | "censored_high" | "nodata"
function [dstar, status, lo, hi] = d_star_interp_row(depths, errs, tol)
    depths = depths(:); errs = errs(:);
    ok = ~isnan(depths) & ~isnan(errs);
    depths = depths(ok); errs = errs(ok);
    [depths, idx] = sort(depths);
    errs = errs(idx);
    if isempty(depths)
        dstar = NaN; status = "nodata"; lo = NaN; hi = NaN; return;
    end
    below = find(errs <= tol, 1, 'first');
    if isempty(below)
        % (b) no crossing at any grid depth: censored high, > s*d_top.
        dstar = NaN; status = "censored_high"; lo = depths(end); hi = NaN; return;
    end
    if below == 1
        % (a) crossing below the grid minimum: right-censored "<= d_min",
        % bracket [d_min/2, d_min] (generalizes the plan's literal "[2,4]",
        % which is what this reduces to when d_min == 4, as it always is on
        % the real grids). The stand-in dstar = d_min makes the resulting
        % overshoot a LOWER BOUND on the true overshoot (pass 10 confirmed
        % this bound direction).
        dstar = depths(1); status = "censored_low"; lo = depths(1) / 2; hi = depths(1); return;
    end
    d_lo = depths(below - 1); e_lo = errs(below - 1);
    d_hi = depths(below);     e_hi = errs(below);
    if e_lo <= 0 || e_hi <= 0 || e_lo <= tol
        dstar = d_hi; status = "ok"; lo = d_lo; hi = d_hi; return;
    end
    tt = (log(tol) - log(e_lo)) / (log(e_hi) - log(e_lo));
    dstar = exp(log(d_lo) + tt * (log(d_hi) - log(d_lo)));
    status = "ok"; lo = d_lo; hi = d_hi;
end

function rows = sel(T, matrix, fname, s, arm)
    rows = strcmp(T.matrix, matrix) & strcmp(T.f, fname) & T.s == s & strcmp(T.arm, arm);
end

function trials = trials_present(T, matrix, fname, s, arm)
    rows = sel(T, matrix, fname, s, arm);
    trials = unique(T.trial(rows));
end

function [depths, errs] = fixed_curve(T, matrix, fname, s, arm, trial)
    rows = sel(T, matrix, fname, s, arm) & T.trial == trial;
    depths = T.d_or_cap(rows);
    errs = T.rel_err(rows);
end

% Per-trial matvecs-to-tol for a FIXED arm: s * d*_interp(tol), with status.
function [mv, status] = fixed_matvecs_to_tol(T, matrix, fname, s, arm, trial, tol)
    [depths, errs] = fixed_curve(T, matrix, fname, s, arm, trial);
    [dstar, status, ~, ~] = d_star_interp_row(depths, errs, tol);
    if status == "censored_high"
        mv = NaN;
    else
        mv = s * dstar;
    end
end

% median [p25 p75] over a numeric vector, NaNs dropped; NaN triple if empty.
function [m, p25, p75] = med_iqr(v)
    v = v(~isnan(v));
    if isempty(v)
        m = NaN; p25 = NaN; p75 = NaN;
    else
        m = median(v); p25 = prctile(v, 25); p75 = prctile(v, 75);
    end
end

function s = fmt_val(m, p25, p75)
    if isnan(m)
        s = '--';
    elseif isnan(p25) || isnan(p75)
        s = sprintf('%.3g', m);
    else
        s = sprintf('%.3g [%.3g, %.3g]', m, p25, p75);
    end
end

% =====================================================================
% [A-priori reachability table -- printed FIRST, per the plan's reading
%  order. Pure post-hoc scan of the already-collected fixed-arm sweep;
%  does not gate or filter any other table (pass 5 F13).]
% =====================================================================
function print_reachability_table(T, matrices, fnames, svals, tols, n)
    fprintf('---- A-priori reachability table ----\n');
    fprintf(['(does not gate any other table; "n/a" = no rows in the CSV for that ' ...
             'cell, e.g. a matrix/f/s/tol slice this run mode did not visit)\n\n']);
    fprintf('%-9s %-7s %-3s %-8s | %-22s %-22s\n', ...
        'matrix', 'f', 's', 'tol', 'block-fixed reaches tol', 'scalar-fixed reaches tol');
    fprintf('%s\n', repmat('-', 1, 80));
    for mi = 1:numel(matrices)
        matrix = matrices{mi};
        for fi = 1:numel(fnames)
            fname = fnames{fi};
            for si = 1:numel(svals)
                s = svals(si);
                for ti = 1:numel(tols)
                    tol = tols(ti);
                    rb = curve_reaches(T, matrix, fname, s, 'block-fixed', tol);
                    rs = curve_reaches(T, matrix, fname, s, 'scalar-fixed', tol);
                    fprintf('%-9s %-7s %-3d %-8.0e | %-22s %-22s\n', ...
                        matrix, fname, s, tol, reach_str(rb), reach_str(rs));
                end
            end
        end
    end
    fprintf('\nblock-fixed wall = floor(n/s): s=4 -> %d, s=16 -> %d\n\n', ...
        block_wall(n, 4), block_wall(n, 16));
end

function s = reach_str(r)
    if isnan(r), s = 'n/a (no data)';
    elseif r,    s = 'YES';
    else,        s = 'no (never <= tol on its grid)';
    end
end

% "reached" = the MEDIAN error-vs-depth curve (median rel_err at each grid
% depth, across whatever trials are present) drops to <= tol at some depth.
function reached = curve_reaches(T, matrix, fname, s, arm, tol)
    rows = sel(T, matrix, fname, s, arm);
    if ~any(rows), reached = NaN; return; end
    depths = T.d_or_cap(rows); errs = T.rel_err(rows);
    ud = unique(depths);
    med = arrayfun(@(d) median(errs(depths == d), 'omitnan'), ud);
    reached = any(med <= tol);
end

% =====================================================================
% [Summary table -- headline slice ONLY: matrix=geo1e6, s=4. One
%  sub-table per (f, tol) -- 4 sub-tables, 5 rows (arms) each.]
% =====================================================================
function print_summary_table(T, viol, fnames, tols, n)
    matrix = 'geo1e6'; s = 4;
    fixed_arms = {'scalar-fixed', 'block-fixed', 'block-fixed-reorth0'};

    fprintf('---- Summary table (headline slice: matrix=%s, s=%d) ----\n', matrix, s);
    for fi = 1:numel(fnames)
        fname = fnames{fi};
        for ti = 1:numel(tols)
            tol = tols(ti);
            fprintf('\n== f=%s, tol=%.0e ==\n', fname, tol);
            fprintf('%-22s | %-24s | %-24s | %-10s | %-10s | %-10s\n', ...
                'arm', 'median matvecs-to-tol', 'certificate overshoot', ...
                'retire %', 'excl(k/m)', 'cert.viol.');
            fprintf('%s\n', repmat('-', 1, 118));

            % ---- fixed arms ----
            for ai = 1:numel(fixed_arms)
                arm = fixed_arms{ai};
                tr = trials_present(T, matrix, fname, s, arm);
                mv = nan(numel(tr), 1);
                for k = 1:numel(tr)
                    mv(k) = fixed_matvecs_to_tol(T, matrix, fname, s, arm, tr(k), tol);
                end
                [m, p25, p75] = med_iqr(mv);
                fprintf('%-22s | %-24s | %-24s | %-10s | %-10s | %-10s\n', ...
                    arm, fmt_val(m, p25, p75), '--', '--', '--', '--');
            end

            % ---- scalar-certified ----
            print_certified_row(T, viol, matrix, fname, s, tol, 'scalar-certified', ...
                'scalar-fixed', 'wall_limited_unused');

            % ---- block-certified ----
            print_certified_row(T, viol, matrix, fname, s, tol, 'block-certified', ...
                'block-fixed', 'wall_limited_unused');
        end
    end
    fprintf('\n');
end

function print_certified_row(T, viol, matrix, fname, s, tol, arm, fixed_oracle_arm, ~)
    rows = sel(T, matrix, fname, s, arm) & T.tol == tol;
    if ~any(rows)
        fprintf('%-22s | %-24s | %-24s | %-10s | %-10s | %-10s\n', arm, '--', '--', '--', '--', '--');
        return;
    end
    sub = T(rows, :);
    m_total = height(sub);
    certified_mask = sub.certified == 1;
    n_excluded = sum(~certified_mask);

    % median matvecs-to-tol: actual matvecs spent, certified rows only.
    [mv_m, mv_p25, mv_p75] = med_iqr(sub.matvecs(certified_mask));

    % overshoot per trial, same-class fixed-arm oracle, same trial, ONLY
    % where the fixed-arm curve status is "ok" (censored_low trials are
    % excluded from the overshoot aggregate per metric 2/3's rule, though
    % they still contributed to F3-left as a bound).
    ov = nan(m_total, 1);
    for k = 1:m_total
        if ~certified_mask(k), continue; end
        trial = sub.trial(k);
        [depths, errs] = fixed_curve(T, matrix, fname, s, fixed_oracle_arm, trial);
        [dstar, status, ~, ~] = d_star_interp_row(depths, errs, tol);
        if status ~= "ok", continue; end
        fixed_mv = s * dstar;
        if fixed_mv > 0
            ov(k) = sub.matvecs(k) / fixed_mv;
        end
    end
    [ov_m, ov_p25, ov_p75] = med_iqr(ov);

    % scalar retirement savings % (scalar-certified row only).
    retire_str = '--';
    if strcmp(arm, 'scalar-certified')
        savings = nan(m_total, 1);
        for k = 1:m_total
            if ~certified_mask(k), continue; end
            smax = sub.col_depth_max(k);
            if smax > 0
                savings(k) = 100 * (s * smax - sub.matvecs(k)) / (s * smax);
            end
        end
        [sv_m, sv_p25, sv_p75] = med_iqr(savings);
        retire_str = fmt_val(sv_m, sv_p25, sv_p75);
    end

    % exclusion count out of m_total (<= 8) trials this slice.
    excl_str = sprintf('%d/%d', n_excluded, m_total);

    % certificate violations (from the metric7 trailer; dash if no entry).
    viol_str = '--';
    for k = 1:numel(viol)
        if strcmp(viol(k).matrix, matrix) && strcmp(viol(k).f, fname) && ...
           viol(k).s == s && abs(viol(k).tol - tol) < 1e-12 && strcmp(viol(k).arm, arm)
            viol_str = sprintf('%d/%d', viol(k).violations, viol(k).checked);
            break;
        end
    end

    fprintf('%-22s | %-24s | %-24s | %-10s | %-10s | %-10s\n', arm, ...
        fmt_val(mv_m, mv_p25, mv_p75), fmt_val(ov_m, ov_p25, ov_p75), ...
        retire_str, excl_str, viol_str);
end

% =====================================================================
% [Okabe-Ito palette + bound-glyph convention, campaign-consistent
%  (see matlab/FunNystromPP_benchmark/plotting/plot_cost_depth.m)]
% =====================================================================
function OI = okabe_ito()
    OI = struct('blue', [0 0.447 0.698], 'sky', [0.337 0.706 0.914], ...
               'green', [0 0.620 0.451], 'orange', [0.902 0.624 0], ...
               'vermillion', [0.835 0.369 0], 'pink', [0.800 0.475 0.655], ...
               'violet', [0.365 0.227 0.608], 'black', [0 0 0]);
end
function bm = bound_marker(own_marker)
    if any(strcmp(own_marker, {'v', '^'}))
        bm = 'x';
    else
        bm = 'v';
    end
end

% =====================================================================
% [F1 -- error-vs-depth, fixed arms only, matrix=geo1e6 ONLY]
% =====================================================================
function make_F1(T, out_dir, n)
    OI = okabe_ito();
    matrix = 'geo1e6';
    fnames = {'sqrt', 'log1p'};
    svals = [4 16];   % row 1 = s=4 (headline), row 2 = s=16
    SER = { 'scalar-fixed',        OI.blue,       'd', '-';
           'block-fixed',          OI.green,      'o', '-';
           'block-fixed-reorth0', OI.vermillion, 's', '--' };

    fig = figure('Color', 'w', 'Position', [20 20 1100 900]);
    tl = tiledlayout(fig, numel(svals), numel(fnames), 'Padding', 'compact', 'TileSpacing', 'compact');
    for si = 1:numel(svals)
        s = svals(si);
        wall = block_wall(n, s);
        for fi = 1:numel(fnames)
            fname = fnames{fi};
            ax = nexttile(tl); hold(ax, 'on');
            set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', 10, 'Box', 'on');
            any_data = false;
            for e = 1:size(SER, 1)
                arm = SER{e, 1};
                rows = sel(T, matrix, fname, s, arm);
                if any(rows)
                    any_data = true;
                    depths = T.d_or_cap(rows); errs = T.rel_err(rows);
                    ud = unique(depths);
                    med = arrayfun(@(d) median(errs(depths == d), 'omitnan'), ud);
                    plot(ax, ud, med, SER{e, 4}, 'Color', SER{e, 2}, 'LineWidth', 1.6, ...
                         'Marker', SER{e, 3}, 'MarkerFaceColor', SER{e, 2}, ...
                         'MarkerSize', 6, 'DisplayName', arm);
                end
            end
            % Wall shading: known n/s value directly, never the wall_limited
            % CSV field (which is block-CERTIFIED-only and NaN on fixed rows).
            yl = ylim(ax); if all(isfinite(yl)) && yl(1) > 0
                xl = xlim(ax);
                xhi = max(xl(2), wall * 1.5);
                patch(ax, [wall xhi xhi wall], [yl(1) yl(1) yl(2) yl(2)], [0.85 0.85 0.85], ...
                      'FaceAlpha', 0.4, 'EdgeColor', 'none', 'HandleVisibility', 'off');
                text(ax, wall, yl(2), ' wall', 'VerticalAlignment', 'top', 'FontSize', 8, 'Color', [0.4 0.4 0.4]);
                ylim(ax, yl);
            end
            grid(ax, 'on');
            xlabel(ax, 'depth d'); ylabel(ax, 'median rel. err');
            title(ax, sprintf('f=%s, s=%d%s', fname, s, ternary(s == 4, ' (headline)', '')), 'FontSize', 11);
            if any_data, legend(ax, 'Location', 'southwest', 'FontSize', 8); end
            if ~any_data
                text(ax, 0.5, 0.5, 'no data in this run', 'Units', 'normalized', ...
                     'HorizontalAlignment', 'center', 'Color', [0.6 0.6 0.6]);
            end
        end
    end
    title(tl, 'F1: error-vs-depth (fixed arms, matrix=geo1e6)', 'FontSize', 13);
    out = fullfile(out_dir, 'F1_error_vs_depth.png');
    exportgraphics(fig, out, 'Resolution', 200);
    close(fig);
    fprintf('  wrote %s\n', out);
end
function r = ternary(cond, a, b)
    if cond, r = a; else, r = b; end
end

% =====================================================================
% [F2 -- matvecs-to-tol by arm across (matrix, s), tol=1e-4, faceted per f]
% =====================================================================
function make_F2(T, out_dir)
    OI = okabe_ito();
    matrices = {'geo1e3', 'geo1e6', 'logu1e6'};
    svals = [4 16];
    tol = 1e-4;
    fnames = {'sqrt', 'log1p'};
    SER = { 'scalar-fixed',      OI.blue,       'd';
           'scalar-certified',   OI.vermillion, 'pentagram';
           'block-fixed',        OI.green,      'o';
           'block-certified',    OI.violet,     '^' };

    groups = {};
    for mi = 1:numel(matrices)
        for si = 1:numel(svals)
            groups{end+1} = sprintf('%s,s=%d', matrices{mi}, svals(si)); %#ok<AGROW>
        end
    end
    nG = numel(groups);
    xoff = linspace(-0.25, 0.25, size(SER, 1));

    fig = figure('Color', 'w', 'Position', [20 20 1300 520]);
    tl = tiledlayout(fig, 1, numel(fnames), 'Padding', 'compact', 'TileSpacing', 'compact');
    for fi = 1:numel(fnames)
        fname = fnames{fi};
        ax = nexttile(tl); hold(ax, 'on');
        set(ax, 'FontSize', 10, 'Box', 'on');
        gi = 0;
        any_data = false;
        for mi = 1:numel(matrices)
            matrix = matrices{mi};
            for si = 1:numel(svals)
                s = svals(si);
                gi = gi + 1;
                for e = 1:size(SER, 1)
                    arm = SER{e, 1};
                    vals = arm_matvecs_at_tol(T, matrix, fname, s, arm, tol);
                    [m, p25, p75] = med_iqr(vals);
                    if isnan(m), continue; end
                    any_data = true;
                    x = gi + xoff(e);
                    lo = max(0, m - p25); hi = max(0, p75 - m);
                    errorbar(ax, x, m, lo, hi, 'LineStyle', 'none', 'Color', SER{e, 2}, ...
                             'CapSize', 4, 'HandleVisibility', 'off');
                    plot(ax, x, m, 'Marker', SER{e, 3}, 'MarkerFaceColor', SER{e, 2}, ...
                         'MarkerEdgeColor', SER{e, 2}, 'MarkerSize', 7, 'LineStyle', 'none', ...
                         'HandleVisibility', 'off');
                end
            end
        end
        for e = 1:size(SER, 1)
            plot(ax, NaN, NaN, 'Marker', SER{e, 3}, 'MarkerFaceColor', SER{e, 2}, ...
                 'MarkerEdgeColor', SER{e, 2}, 'LineStyle', 'none', 'DisplayName', SER{e, 1});
        end
        set(ax, 'XTick', 1:nG, 'XTickLabel', groups, 'XTickLabelRotation', 30);
        xlim(ax, [0.5, nG + 0.5]);
        grid(ax, 'on');
        ylabel(ax, 'matvecs to reach tol=1e-4');
        title(ax, sprintf('f=%s', fname), 'FontSize', 11);
        legend(ax, 'Location', 'northoutside', 'NumColumns', 4, 'FontSize', 8);
        if ~any_data
            text(ax, 0.5, 0.5, 'no data at tol=1e-4 in this run', 'Units', 'normalized', ...
                 'HorizontalAlignment', 'center', 'Color', [0.6 0.6 0.6]);
        end
        set(ax, 'YScale', 'log');
    end
    title(tl, 'F2: matvecs-to-tol by arm across (matrix, s), tol=1e-4', 'FontSize', 13);
    out = fullfile(out_dir, 'F2_matvecs_to_tol.png');
    exportgraphics(fig, out, 'Resolution', 200);
    close(fig);
    fprintf('  wrote %s\n', out);
end

% Per-trial matvecs-to-tol for either a fixed arm (s*d*_interp) or a
% certified arm (actual matvecs, certified rows only), as a vector over trials.
function vals = arm_matvecs_at_tol(T, matrix, fname, s, arm, tol)
    if any(strcmp(arm, {'scalar-fixed', 'block-fixed', 'block-fixed-reorth0'}))
        tr = trials_present(T, matrix, fname, s, arm);
        vals = nan(numel(tr), 1);
        for k = 1:numel(tr)
            vals(k) = fixed_matvecs_to_tol(T, matrix, fname, s, arm, tr(k), tol);
        end
    else
        rows = sel(T, matrix, fname, s, arm) & T.tol == tol & T.certified == 1;
        vals = T.matvecs(rows);
    end
end

% =====================================================================
% [F3 -- certificate story, s=4 headline slice]
% =====================================================================
function make_F3(T, out_dir, viol) %#ok<INUSD>
    OI = okabe_ito();
    s = 4;
    matrices = {'geo1e3', 'geo1e6', 'logu1e6'};
    fnames = {'sqrt', 'log1p'};
    tols = [1e-2 1e-4];

    fig = figure('Color', 'w', 'Position', [20 20 1300 560]);
    tl = tiledlayout(fig, 1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

    % ---- left: overshoot factor, grouped by (f,tol), 2 series (scalar/block
    %      certified), 3 matrix-dots per group, point + grid-bracket error bar.
    ax = nexttile(tl); hold(ax, 'on');
    set(ax, 'YScale', 'log', 'FontSize', 10, 'Box', 'on');
    groups = {};
    for fi = 1:numel(fnames)
        for ti = 1:numel(tols)
            groups{end+1} = sprintf('%s,%.0e', fnames{fi}, tols(ti)); %#ok<AGROW>
        end
    end
    SER3 = {'scalar-certified', OI.vermillion, 'pentagram', 'scalar-fixed';
           'block-certified',  OI.violet,     '^',         'block-fixed'};
    mjit = linspace(-0.15, 0.15, numel(matrices));
    n_censored_low_pts = 0; n_absent = 0; any_left = false;
    gi = 0;
    for fi = 1:numel(fnames)
        fname = fnames{fi};
        for ti = 1:numel(tols)
            tol = tols(ti);
            gi = gi + 1;
            for e = 1:size(SER3, 1)
                arm = SER3{e, 1};
                fixed_oracle_arm = SER3{e, 4};
                for mi = 1:numel(matrices)
                    matrix = matrices{mi};
                    rows = sel(T, matrix, fname, s, arm) & T.tol == tol & T.certified == 1;
                    if ~any(rows), continue; end
                    sub = T(rows, :);
                    ovs = nan(height(sub), 1); is_bound = false(height(sub), 1);
                    lo_b = nan(height(sub), 1); hi_b = nan(height(sub), 1);
                    for k = 1:height(sub)
                        trial = sub.trial(k);
                        [depths, errs] = fixed_curve(T, matrix, fname, s, fixed_oracle_arm, trial);
                        [dstar, status, lo, hi] = d_star_interp_row(depths, errs, tol);
                        if status == "censored_high" || status == "nodata", continue; end
                        fixed_mv = s * dstar;
                        if fixed_mv <= 0, continue; end
                        ovs(k) = sub.matvecs(k) / fixed_mv;
                        is_bound(k) = (status == "censored_low");
                        lo_b(k) = s * lo; hi_b(k) = s * hi;
                    end
                    keep = ~isnan(ovs);
                    if ~any(keep), continue; end
                    ovs = ovs(keep); is_bound = is_bound(keep);
                    m = median(ovs); % per (matrix,f,tol,arm): one dot (median over its trials)
                    x = gi + mjit(mi);
                    any_left = true;
                    if any(is_bound)
                        n_censored_low_pts = n_censored_low_pts + 1;
                        plot(ax, x, m, 'Marker', bound_marker(SER3{e, 3}), 'MarkerFaceColor', 'none', ...
                             'MarkerEdgeColor', SER3{e, 2}, 'LineWidth', 1.3, 'MarkerSize', 7, ...
                             'LineStyle', 'none', 'HandleVisibility', 'off');
                    else
                        loB = median(lo_b(~isnan(lo_b)), 'omitnan');
                        hiB = median(hi_b(~isnan(hi_b)), 'omitnan');
                        if ~isnan(loB) && ~isnan(hiB)
                            errorbar(ax, x, m, max(0, m - loB), max(0, hiB - m), ...
                                     'LineStyle', 'none', 'Color', SER3{e, 2}, 'CapSize', 4, ...
                                     'HandleVisibility', 'off');
                        end
                        plot(ax, x, m, 'Marker', SER3{e, 3}, 'MarkerFaceColor', SER3{e, 2}, ...
                             'MarkerEdgeColor', SER3{e, 2}, 'MarkerSize', 7, 'LineStyle', 'none', ...
                             'HandleVisibility', 'off');
                    end
                end
            end
        end
    end
    for e = 1:size(SER3, 1)
        plot(ax, NaN, NaN, 'Marker', SER3{e, 3}, 'MarkerFaceColor', SER3{e, 2}, ...
             'MarkerEdgeColor', SER3{e, 2}, 'LineStyle', 'none', 'DisplayName', SER3{e, 1});
    end
    set(ax, 'XTick', 1:numel(groups), 'XTickLabel', groups, 'XTickLabelRotation', 20);
    xlim(ax, [0.5, numel(groups) + 0.5]);
    grid(ax, 'on'); ylabel(ax, 'certificate overshoot (certified / fixed-oracle matvecs)');
    title(ax, 'certificate overshoot (s=4 headline)', 'FontSize', 11);
    legend(ax, 'Location', 'northoutside', 'NumColumns', 2, 'FontSize', 8);
    if ~any_left
        text(ax, 0.5, 0.5, 'no data in this run', 'Units', 'normalized', ...
             'HorizontalAlignment', 'center', 'Color', [0.6 0.6 0.6]);
    end
    fprintf('  F3-left: %d hollow-bound (censored_low) points plotted; %d group/arm cells absent (no crossing data)\n', ...
        n_censored_low_pts, n_absent);

    % ---- right: block d_stop vs scalar max_j t_used, color = savings %.
    ax2 = nexttile(tl); hold(ax2, 'on'); set(ax2, 'FontSize', 10, 'Box', 'on');
    xs = []; ys = []; cs = [];
    for fi = 1:numel(fnames)
        fname = fnames{fi};
        for ti = 1:numel(tols)
            tol = tols(ti);
            for mi = 1:numel(matrices)
                matrix = matrices{mi};
                brows = sel(T, matrix, fname, s, 'block-certified') & T.tol == tol & T.certified == 1;
                srows = sel(T, matrix, fname, s, 'scalar-certified') & T.tol == tol & T.certified == 1;
                bt = T(brows, :); st = T(srows, :);
                for k = 1:height(bt)
                    trial = bt.trial(k);
                    sr = st(st.trial == trial, :);
                    if isempty(sr), continue; end
                    d_stop = bt.d_or_cap(k);
                    t_max = sr.col_depth_max(1);
                    savings = 100 * (s * t_max - sr.matvecs(1)) / (s * t_max);
                    xs(end+1) = t_max; ys(end+1) = d_stop; cs(end+1) = savings; %#ok<AGROW>
                end
            end
        end
    end
    if ~isempty(xs)
        scatter(ax2, xs, ys, 55, cs, 'filled', 'MarkerEdgeColor', [0.3 0.3 0.3]);
        cb = colorbar(ax2); cb.Label.String = 'scalar retirement savings %';
        colormap(ax2, parula);
        lim = [0, max([xs(:); ys(:)]) * 1.1 + 1];
        plot(ax2, lim, lim, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
        xlim(ax2, lim); ylim(ax2, lim);
    else
        text(ax2, 0.5, 0.5, 'no certified (block,scalar) trial pairs in this run', ...
             'Units', 'normalized', 'HorizontalAlignment', 'center', 'Color', [0.6 0.6 0.6]);
    end
    grid(ax2, 'on');
    xlabel(ax2, 'scalar: max_j t_used[j]'); ylabel(ax2, 'block: d_stop');
    title(ax2, 'certificate-firing comparison (dashed = y=x)', 'FontSize', 11);

    title(tl, 'F3: certificate story (s=4 headline)', 'FontSize', 13);
    out = fullfile(out_dir, 'F3_certificate_story.png');
    exportgraphics(fig, out, 'Resolution', 200);
    close(fig);
    fprintf('  wrote %s\n', out);
end
