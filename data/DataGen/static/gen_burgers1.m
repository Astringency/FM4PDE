function output_path = gen_burgers1(dataset_type, N, s, steps, out_root, seed, overwrite)
    if nargin < 1
        error('dataset_type is required: train, easytest, or hardtest');
    end
    if nargin < 2 || isempty(N)
        N = 10000;
    end
    if nargin < 3 || isempty(s)
        s = 128;
    end
    if nargin < 4 || isempty(steps)
        steps = 127;
    end
    if nargin < 5 || isempty(out_root)
        out_root = '/large_storage/zhangxf/PDEdata';
    end

    profile = get_generation_profile('burgers', dataset_type);
    if nargin < 6 || isempty(seed)
        seed = profile.seed_offset;
    end
    if nargin < 7 || isempty(overwrite)
        overwrite = false;
    end
    rng(seed, 'twister');

    dataset_type = profile.dataset_type;
    grf_gamma = profile.alpha;
    grf_tau = profile.tau;
    grf_sigma = grf_tau^(grf_gamma - 0.5);
    generation_seed = seed;
    viscosity = 1 / 100;

    output_dir = fullfile(char(out_root), 'burgers');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    output_path = fullfile(output_dir, sprintf( ...
        'burger_%s_%d-%d-%d.mat', dataset_type, N, s, steps + 1));
    if exist(output_path, 'file') && ~overwrite
        error('Output already exists: %s. Enable overwrite to replace it.', output_path);
    end

    input = zeros(N, s);
    output = zeros(N, steps + 1, s);
    tspan = linspace(0, 1, steps + 1);
    x = linspace(0, 1, s + 1);

    t0 = tic;
    updateEvery = max(1, floor(N / 100));

    for sample_idx = 1:N
        u0 = GRF1(s / 2, 0, grf_gamma, grf_tau, grf_sigma, "periodic");
        u = burgers1(u0, tspan, s, viscosity);

        u0eval = u0(x);
        input(sample_idx, :) = u0eval(1:end-1);
        output(sample_idx, 1, :) = input(sample_idx, :);
        for time_idx = 2:(steps + 1)
            output(sample_idx, time_idx, :) = u{time_idx}.values;
        end

        if mod(sample_idx, updateEvery) == 0 || sample_idx == N
            elapsed = toc(t0);
            eta = elapsed * (N - sample_idx) / sample_idx;
            fprintf('\rProgress [Burgers]: %6.2f%%  [%d/%d]  elapsed: %.1fs  ETA: %.1fs', ...
                100 * sample_idx / N, sample_idx, N, elapsed, eta);
        end
    end

    save(output_path, 'output', 'input', 'tspan', 'dataset_type', ...
        'grf_gamma', 'grf_tau', 'grf_sigma', 'generation_seed', ...
        'viscosity');
    fprintf('\nSaved Burgers %s data to %s\n', dataset_type, output_path);
end
