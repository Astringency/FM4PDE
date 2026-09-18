function output_path = generate_poisson(dataset_type, N, S, out_root, seed, overwrite, shard_id)
    if nargin < 1
        error('dataset_type is required: train, id, smooth, rough, rough2, or rough3');
    end
    profile = get_generation_profile('poisson', dataset_type);
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

    output_dir = fullfile(char(out_root), 'poisson');
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    if strcmp(dataset_type, 'train')
        output_name = sprintf('poisson_%d-%d-%d_%d.mat', N, S, S, shard_id);
    else
        output_name = sprintf('poisson_test_%d-%d-%d_%s.mat', N, S, S, dataset_type);
    end
    output_path = fullfile(output_dir, output_name);
    if exist(output_path, 'file') && ~overwrite
        fprintf('Skipping existing Poisson file: %s\n', output_path);
        return;
    end
    
        % Preallocate arrays
        f_data = zeros(N, S, S);
        phi_data = zeros(N, S, S);
    
        t0 = tic;
        updateEvery = max(1, floor(N / 100));
    
        for i = 1:N
            % Generate the coefficient f using GRF
            f = GRF(grf_alpha, grf_tau, S);
            
            % Solve the Poisson equation for phi
            phi = solve_poisson(f, S);
            
            % Store the generated data
            f_data(i, :, :) = f;
            phi_data(i, :, :) = phi;

            if mod(i, updateEvery) == 0 || i == N
                elapsed = toc(t0);
                eta = elapsed * (N - i) / i;

                fprintf('\rProgress [Poisson]: %6.2f%%  [%d/%d]  elapsed: %.1fs  ETA: %.1fs', ...
                    100 * i / N, i, N, elapsed, eta);
            end
        end
    
    save(output_path, 'f_data', 'phi_data', 'dataset_type', 'grf_alpha', ...
        'grf_tau', 'generation_seed', 'shard_id');
    fprintf('\nSaved Poisson %s data to %s\n', dataset_type, output_path);
end

function phi = solve_poisson(f, S)
    % Assuming f is already on a SxS grid and represents the source term
    % uniformly distributed across the domain [0,1]x[0,1].
    
    % Define grid spacing
    h = 1 / (S - 1);
    
    % Initialize phi
    phi = zeros(S, S);
    
    % Assemble the system matrix for the Poisson equation
    % For simplicity, using a 5-point Laplacian stencil with Dirichlet boundary conditions
    N = S^2; % Total number of points in the grid
    A = sparse(N, N);
    B = reshape(f, [N, 1]); % Reshape f into a vector for the linear system
    
    for i = 1:S
        for j = 1:S
            index = (i - 1) * S + j; % Convert (i, j) to linear index
            
            % Apply Dirichlet boundary conditions: phi = 0 at boundaries
            if i == 1 || i == S || j == 1 || j == S
                A(index, index) = 1; % Boundary points
                B(index) = 0; % Assuming phi = 0 on the boundary
            else
                % Internal points - Discretize Laplacian operator
                A(index, index) = -4 / h^2;
                A(index, index - 1) = 1 / h^2; % Left
                A(index, index + 1) = 1 / h^2; % Right
                A(index, index - S) = 1 / h^2; % Below
                A(index, index + S) = 1 / h^2; % Above
            end
        end
    end
    
    % Solve the linear system A*phi = B
    phi_vec = A \ B;
    
    % Reshape the solution back to a 2D grid
    phi = reshape(phi_vec, [S, S]);
end
