
% Define the path to the CSV file
csv_file = 'small_features.csv';  % Replace with the actual path if needed

% Read the CSV file as a table
data_table = readtable(csv_file, 'Delimiter', ';', 'TextType', 'string');

% Extract image paths
image_paths = data_table.ImagePath;

% Extract feature vectors and decode JSON strings to numerical arrays
num_samples = height(data_table);  % Get the number of rows
feature_vectors = cell(num_samples, 1);  % Preallocate a cell array

for i = 1:num_samples
    feature_vectors{i} = jsondecode(data_table.FeatureVector(i));  % Convert JSON string to numeric array
end

% Convert cell array to matrix (assuming all feature vectors have the same length)
feature_matrix = cell2mat(feature_vectors.').';

% Display information
disp("Feature extraction completed!");
disp("Number of images loaded: " + num_samples);
disp("Feature matrix size: " + mat2str(size(feature_matrix)));

N = min(num_samples, 15);
start = 2;
X = feature_matrix(1+start:N+start,:).';

Z = gsp_distanz(X).^2;


k = 4;
theta = gsp_compute_graph_learning_theta(Z, k);

t1 = tic;
[W, info_1] = gsp_learn_graph_log_degrees(theta * Z, 1, 1);
t1 = toc(t1);
W(W<1e-2) = 0;


G = gsp_ring(N);

G = gsp_update_weights(G, W);
figure; gsp_plot_graph(G);



scale = .1;

l = .5 - scale/2;
for i = 1:N
    axes('Position',[l+.4*cos(2*pi/N*(i-1)), l-.4*sin(2*pi/N*(i-1)) ,scale, scale])
    imshow(image_paths(i+start))
end

exportgraphics(gcf, "results/imagenet-graph-n.pdf","Append", false)


figure; imagesc(W); title(['Average edges/node: ', num2str(nnz(W)/G.N)]);
exportgraphics(gcf, "results/imagenet-w-n.pdf","Append", false)



