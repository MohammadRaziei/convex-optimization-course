gsp_reset_seed(1);
G = gsp_2dgrid(16);
n = G.N;
W0 = G.W;
G = gsp_graph(G.W, G.coords+.05*rand(size(G.coords)));
figure; gsp_plot_graph(G);

X = G.coords';

% First compute the matrix of all pairwise distances Z
Z = gsp_distanz(X).^2;
% this is a highly dense matrix
figure; imagesc(Z);
% Let's learn a graph, we need to know two parameters a and b according to
% [1, Kalofolias, 2016]
a = 1;
b = 1;
W = gsp_learn_graph_log_degrees(Z, a, b);
% this matrix is naturally sparse, but still has many connections unless we
% "play" with the parameters a and b to make it sparser
figure; imagesc(W);
% update weights
W(W<1e-4) = 0;
G = gsp_update_weights(G, W);

% suppose we want 4 edges per node on average
k = 4;
theta = gsp_compute_graph_learning_theta(Z, k);
% then instead of playing with a and b, we k keep them equal to 1 and
% multiply Z by theta for graph learning:
t1 = tic;
[W, info_1] = gsp_learn_graph_log_degrees(theta * Z, 1, 1);
t1 = toc(t1);
% clean edges close to zero
W(W<1e-4) = 0;
% indeed, the new adjacency matrix has around 4 non zeros per row!
figure; imagesc(W); title(['Average edges/node: ', num2str(nnz(W)/G.N)]);
% update weights and plot
G = gsp_update_weights(G, W);
figure; gsp_plot_graph(G);