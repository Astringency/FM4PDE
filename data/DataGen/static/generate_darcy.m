function output_path = generate_darcy(dataset_type, N, S, out_root, seed, overwrite)
    if nargin < 1
        error('dataset_type is required: train, easytest, or hardtest');
    end
    if nargin < 2 || isempty(N)
        N = 10000;
    end
    if nargin < 3 || isempty(S)
        S = 128;
    end
    if nargin < 4 || isempty(out_root)
        out_root = '/large_storage/zhangxf/PDEdata';
    end

    profile = get_generation_profile('darcy', dataset_type);
    if nargin < 5 || isempty(seed)
        seed = profile.seed_offset;
    end
    if nargin < 6 || isempty(overwrite)
        overwrite = false;
    end
    rng(seed, 'twister');

    dataset_type = profile.dataset_type;
    grf_alpha = profile.alpha;
    grf_tau = profile.tau;
    generation_seed = seed;

    output_dir = fullfile(char(out_root), 'darcy');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    output_path = fullfile(output_dir, sprintf( ...
        'darcy_%s_%d-%d-%d.mat', dataset_type, N, S, S));
    if exist(output_path, 'file') && ~overwrite
        error('Output already exists: %s. Enable overwrite to replace it.', output_path);
    end
    
        % Preallocate arrays to store the generated data
        lognorm_a_data = zeros(N, S, S);
        thresh_a_data = zeros(N, S, S);
        lognorm_p_data = zeros(N, S, S);
        thresh_p_data = zeros(N, S, S);
    
        % Forcing function, f(x) = 1
        f = ones(S, S);

        t0 = tic;
        updateEvery = max(1, floor(N / 100));
    
        for i = 1:N
            % Generate random coefficients from N(0,C)
            norm_a = GRF(grf_alpha, grf_tau, S);
    
            % Exponentiate it to ensure a(x) > 0 (Lognormal)
            lognorm_a = exp(norm_a);
    
            % Thresholding to achieve ellipticity
            thresh_a = zeros(S, S);
            thresh_a(norm_a >= 0) = 12;
            thresh_a(norm_a < 0) = 4;
    
            % Solve PDE: -div(a(x)*grad(p(x))) = f(x)
            lognorm_p = solve_gwf(lognorm_a, f);
            thresh_p = solve_gwf(thresh_a, f);
    
            % Store the generated data
            lognorm_a_data(i, :, :) = lognorm_a;
            thresh_a_data(i, :, :) = thresh_a;
            lognorm_p_data(i, :, :) = lognorm_p;
            thresh_p_data(i, :, :) = thresh_p;

            if mod(i, updateEvery) == 0 || i == N
                elapsed = toc(t0);
                eta = elapsed * (N - i) / i;

                fprintf('\rProgress [Darcy]: %6.2f%%  [%d/%d]  elapsed: %.1fs  ETA: %.1fs', ...
                    100 * i / N, i, N, elapsed, eta);
            end
        end
    
    save(output_path, 'lognorm_a_data', 'thresh_a_data', 'lognorm_p_data', ...
        'thresh_p_data', 'dataset_type', 'grf_alpha', 'grf_tau', ...
        'generation_seed', '-v7.3');
    fprintf('\nSaved Darcy %s data to %s\n', dataset_type, output_path);
end
