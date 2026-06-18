% for round = 1:5
for round = 6
    rng(round, "twister");
    sprintf("Init Round %d", round)
    % number of realizations to generate
    N = 10000;
    
    % parameters for the Gaussian random field
    gamma = 2.5;
    tau = 7;
    sigma = 7^(2);
    
    % viscosity
    visc = 1/100;
    
    % grid size
    s = 128;
    steps = 127;
    
    
    input = zeros(N, s);
    if steps == 1
        output = zeros(N, s);
    else
        output = zeros(N, steps, s);
    end
    
    tspan = linspace(0,1,steps+1);
    x = linspace(0,1,s+1);

    t0 = tic;
    updateEvery = max(1, floor(N / 100));

    for j=1:N
        u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
        u = burgers1(u0, tspan, s, visc);
        
        u0eval = u0(x);
        input(j,:) = u0eval(1:end-1);
        
        if steps == 1
            output(j,:) = u.values;
        else
            for k=2:(steps+1)
                output(j,k,:) = u{k}.values;
            end
        end
        
        output(j,1,:)=input(j,:);

        if mod(j, updateEvery) == 0 || j == N
            elapsed = toc(t0);
            eta = elapsed * (N - j) / j;

            fprintf('\rProgress [Darcy]: %6.2f%%  [%d/%d]  elapsed: %.1fs  ETA: %.1fs', ...
                100 * j / N, j, N, elapsed, eta);
        end
    end
    
    sprintf("Saving (Round %d)", round)

    output = output(:, 1:end, :);
    tspan = tspan(1:end);
    x = x(1:end);

    % filename = sprintf('/large_storage/zhangxf/PDEdata/burgers/burger_%d-%d-%d_%d.mat', N, s, steps+1, round);
    filename = sprintf('/large_storage/zhangxf/PDEdata/burgers/burger_test_%d-%d-%d_%d.mat', N, s, steps+1, round);
    save(filename, 'output', 'input');

end
