function generateHelmholtztestData(N, S, k)
% function generateHelmholtztestData(N, S)
    % for round = 1:5
    for round = 6
    % for k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if nargin < 1
            % N = 10000; % Default number of generations
            N = 1000; % Default number of generations
        end
        if nargin < 2
            S = 128; % Default resolution
        end
        if nargin < 3
            k = 1; % Default k
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
        
        % Parameters for GRF
        % alpha = 2;
        % tau = 3;
        alpha = 3;
        tau = 4;
        
        hbar = waitbar(0, 'Generating...');

        for i = 1:N
            waitbar(i/N, hbar, sprintf('Generating Inhom Helmholtz: %d', i));
            f = GRF(alpha, tau, S);
            f_data(i, :, :) = f;
            f(1, :) = 0; f(S, :) = 0; f(:, 1) = 0; f(:, S) = 0;
            
            A = L_full + k^2 * speye(S^2);
            f_vector = reshape(f, [S^2, 1]);
            psi_vector = A \ f_vector;
            psi = reshape(psi_vector, [S, S]);
            
            psi_data(i, :, :) = psi;
            
        end
        delete(hbar);
        % Save the dataset
        if ~exist('data', 'dir')
            mkdir('data');
        end
        % filename = sprintf('/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_%d-%d-%d_%d.mat', N, S, S, round);
        % filename = sprintf('/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_%d-%d-%d_test.mat', N, S, S);
        filename = sprintf('/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_%d-%d-%d_test_2.mat', N, S, S);
        % filename = sprintf('/large_storage/zhangxf/PDEdata/helmholtz/helmholtz_%d-%d-%d_k%d.mat', N, S, S, k);
        save(filename, 'f_data', 'psi_data');
    end
end
