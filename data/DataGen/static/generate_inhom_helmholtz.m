function output_path = generate_inhom_helmholtz(dataset_type, N, S, k, out_root, seed, overwrite, shard_id)
    if nargin < 1
        error('dataset_type is required: train, id, smooth, or rough');
    end
    if nargin < 2 || isempty(N)
        N = 10000;
    end
    if nargin < 3 || isempty(S)
        S = 128;
    end
    if nargin < 4 || isempty(k)
        k = 1;
    end
    if nargin < 5 || isempty(out_root)
        out_root = '/large_storage/zhangxf/PDEdata';
    end

    profile = get_generation_profile('helmholtz', dataset_type);
    if nargin < 6 || isempty(seed)
        seed = profile.seed_offset;
    end
    if nargin < 7 || isempty(overwrite)
        overwrite = false;
    end
    if nargin < 8 || isempty(shard_id)
        shard_id = 1;
    end
    rng(seed, 'twister');

    dataset_type = profile.dataset_type;
    grf_alpha = profile.alpha;
    grf_tau = profile.tau;
    generation_seed = seed;

    output_dir = fullfile(char(out_root), 'helmholtz');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    if strcmp(dataset_type, 'train')
        output_name = sprintf('helmholtz_%d-%d-%d_%d.mat', N, S, S, shard_id);
    else
        output_name = sprintf('helmholtz_test_%d-%d-%d_%s.mat', N, S, S, dataset_type);
    end
    output_path = fullfile(output_dir, output_name);
    if exist(output_path, 'file') && ~overwrite
        fprintf('Skipping existing Helmholtz file: %s\n', output_path);
        return;
    end

        f_data = zeros(N, S, S);
        psi_data = zeros(N, S, S);
        
        h = 1 / (S - 1);  
        x = linspace(0, 1, S);
        y = linspace(0, 1, S);
        [X, Y] = meshgrid(x, y);
        
        e = ones(S, 1);
        L = spdiags([e -2*e e], -1:1, S, S) / h^2;

        L(1, :) = 0; L(1, 1) = 1; 
        L(S, :) = 0; L(S, S) = 1; 
        L_full = kron(speye(S), L) + kron(L, speye(S));
        
        t0 = tic;
        updateEvery = max(1, floor(N / 100));
        
        for i = 1:N
            f = GRF(grf_alpha, grf_tau, S);
            f_data(i, :, :) = f;
            f(1, :) = 0; f(S, :) = 0; f(:, 1) = 0; f(:, S) = 0;
            
            A = L_full + k^2 * speye(S^2);
            f_vector = reshape(f, [S^2, 1]);
            psi_vector = A \ f_vector;
            psi = reshape(psi_vector, [S, S]);
            
            psi_data(i, :, :) = psi;
            
            if mod(i, updateEvery) == 0 || i == N
                elapsed = toc(t0);
                eta = elapsed * (N - i) / i;

                fprintf('\rProgress [Helmholtz]: %6.2f%%  [%d/%d]  elapsed: %.1fs  ETA: %.1fs', ...
                    100 * i / N, i, N, elapsed, eta);
            end
        end

    save(output_path, 'f_data', 'psi_data', 'dataset_type', 'grf_alpha', ...
        'grf_tau', 'generation_seed', 'shard_id', 'k');
    fprintf('\nSaved Helmholtz %s data to %s\n', dataset_type, output_path);
end
