function generate_burgers_from_initial(input_file, output_file, chebfun_root)
% Replay the original periodic SPIN generator from sampled initial states.
addpath(chebfun_root);
addpath(fileparts(mfilename('fullpath')));
d = load(input_file);
assert(size(d.initial, 2) == 128);
tspan = linspace(0, 1, 128);
trajectory = zeros(size(d.initial, 1), 128, 128);
seconds = zeros(size(d.initial, 1), 1);
for i = 1:size(d.initial, 1)
    tic;
    initial = chebfun(d.initial(i, :).', [0 1], 'trig');
    result = burgers1(initial, tspan, 128, 0.01);
    trajectory(i, 1, :) = d.initial(i, :);
    for t = 2:128
        trajectory(i, t, :) = result{t}.values;
    end
    seconds(i) = toc;
    assert(all(isfinite(trajectory(i, :, :)), 'all'));
    if mod(i, 20)==0 || i==size(d.initial, 1)
        fprintf('BURGERS SOLVE %d/%d %.3f sec/sample\n', i, size(d.initial, 1), mean(seconds(1:i)));
    end
end
save(output_file, 'trajectory', 'seconds', '-v7');
end
