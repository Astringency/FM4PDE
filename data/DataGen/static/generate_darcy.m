function output_path = generate_darcy(dataset_type, N, S, out_root, seed, overwrite, shard_id)
    if nargin < 1
        error('dataset_type is required: train, id, smooth, rough, rough2, or rough3');
    end
    profile = get_generation_profile('darcy', dataset_type);
    if nargin < 2 || isempty(N)
        N = profile.default_samples;
    end
    if nargin < 3 || isempty(S)
        S = 128;
    end
    if nargin < 4 || isempty(out_root)
        out_root = '/large_storage/zhangxf/PDEdata';
    end

    if nargin < 5 || isempty(seed)
        seed = profile.seed_offset;
    end
    if nargin < 6 || isempty(overwrite)
        overwrite = false;
    end
    if nargin < 7 || isempty(shard_id)
        shard_id = 1;
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
    if strcmp(dataset_type, 'train')
        output_name = sprintf('darcy_%d-%d-%d_%d.mat', N, S, S, shard_id);
    else
        output_name = sprintf('darcy_test_%d-%d-%d_%s.mat', N, S, S, dataset_type);
    end
    output_path = fullfile(output_dir, output_name);
    if exist(output_path, 'file') && ~overwrite
        fprintf('Skipping existing Darcy file: %s\n', output_path);
        return;
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
        'generation_seed', 'shard_id', '-v7.3');
    fprintf('\nSaved Darcy %s data to %s\n', dataset_type, output_path);
end
