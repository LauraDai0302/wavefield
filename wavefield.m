% Parameters
kb = 100; % Wave number
lambda = 2 * pi / kb; % Wavelength
x_limit = lambda; % Domain limits
y_limit = lambda;

% Source position
xs = lambda / 2;
ys = 2 * lambda;

% Grid definition
h = lambda / 20; % Grid step size
x = 0:h:x_limit-h;
y = 0:h:y_limit-h;
[X, Y] = meshgrid(x, y);

[m, n] = size(X);


figure;
scatter(X(:), Y(:), 'filled');
title('Grid Points');
xlabel('X');
ylabel('Y');
axis equal;
grid on;
rho = X + 1i * Y; % Complex position vector rho
x_inc_mid = x + h/2;
y_inc_mid = y + h/2;

% Compute number of grid points
Nx = length(x);
Ny = length(y);
N = Nx * Ny;

disp(['Total number of grid points N = ', num2str(N)]);

% Incident field calculation
rho_s = xs + 1i * ys; % Source position in complex form
k_rho = kb * abs(rho - rho_s); % Distance from source to each grid point
u_inc = -1j * (1/4) * besselh(0, 2, k_rho); % Incident field

% Extract real and imaginary parts
u_real = real(u_inc);
u_imag = imag(u_inc);
u_abs = abs(u_inc);

% Plotting
figure;

% Real part
subplot(1, 3, 1);
imagesc(x, y, u_real);
%title('Real Part');
xlabel('x');
ylabel('y');
xlim([0 6]);  
ylim([0 6]);  
colorbar;
axis equal tight;
title(['Real Part of $$\hat{u}_{inc}$$'], 'Interpreter', 'latex', 'FontSize', 14, 'HorizontalAlignment', 'center');

% Imaginary part
subplot(1, 3, 2);
imagesc(x, y, u_imag);
%title('Imaginary Part of $$\hat{u}_{inc}$$''Interpreter', 'tex');
xlabel('x');
ylabel('y');
xlim([0 6]);  
ylim([0 6]);  
colorbar;
axis equal tight;
title(['Imaginary Part of $$\hat{u}_{inc}$$'], 'Interpreter', 'latex', 'FontSize', 14, 'HorizontalAlignment', 'center');

% Absolute value
subplot(1, 3, 3);
imagesc(x, y, u_abs);
xlabel('x');
ylabel('y');
xlim([0 6]);  
ylim([0 6]);  
colorbar;
axis equal tight;
title(['Absolute Value of $$\hat{u}_{inc}$$'], 'Interpreter', 'latex', 'FontSize', 14, 'HorizontalAlignment', 'center');
% Ensure proper orientation
set(gca, 'YDir', 'reverse');

% Adjust layout
sgtitle('Incident Field Visualization');

% Display the plot
colormap('jet'); % Choose colormap
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555555
% Plotting Configuration Sketch
figure;
hold on;

% Define object domain D
lambda = 2 * pi / kb; % Wavelength
x_limit = lambda; % Domain limits
y_limit = lambda;

% Object domain boundaries
rectangle('Position', [0, 0, lambda, lambda], 'EdgeColor', 'k');
% Object domain boundaries
Dobj = [0, 0, lambda, lambda]; % [x_min, y_min, width, height]


% Source position
plot(real(rho_s), imag(rho_s), 'ro', 'MarkerFaceColor', 'r'); % Source position
text(real(rho_s), imag(rho_s), ' Source Position', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right'); % Source position annotation

hold off;

title('Configuration Sketch');
xlabel('x');
ylabel('y');
axis equal;
set(gca, 'YDir', 'reverse');
xlim([0, 60]);
% %% % kb = 2
% kb_doubled = 2;
% 
% u_hat_inc_double = zeros(n, n);
% for x = 1:n
%     for y = 1:n
%         u_hat_inc_double(y, x) = -0.25i*besselh(0, 2, kb_doubled*norm([x_inc_mid(x)-xs, y_inc_mid(y)-ys]));
%     end
% end
% 
% u_hat_inc_real_double = real(u_hat_inc_double);
% u_hat_inc_imag_double = imag(u_hat_inc_double);
% 
% figure()
% % imshow(u_hat_inc_real_double, [], 'InitialMagnification','fit');
% % imshow(u_hat_inc_imag_double, [], 'InitialMagnification','fit');
% imagesc(x_inc_mid, y_inc_mid, u_hat_inc_real_double);
% axis equal tight
% title('kb = 2, real')
% imagesc(x_inc_mid, y_inc_mid, u_hat_inc_imag_double);
% axis equal tight
% title('kb = 2, imag')
% axis equal tight

% %% Change the location of the source
% % Parameters
% kb = 2; % Wave number
% lambda = 2 * pi / kb; % Wavelength
% x_limit = lambda; % Domain limits
% y_limit = lambda;
% 
% % Source position
% xs = lambda / 2;
% ys = 10*lambda;
% 
% % Grid definition
% h = lambda / 20; % Grid step size
% x = 0:h:x_limit;
% y = 0:h:y_limit;
% [X, Y] = meshgrid(x, y);
% rho = X + 1i * Y; % Complex position vector rho
% 
% % Compute number of grid points
% Nx = length(x);
% Ny = length(y);
% N = Nx * Ny;
% 
% disp(['Total number of grid points N = ', num2str(N)]);
% 
% % Incident field calculation
% rho_s = xs + 1i * ys; % Source position in complex form
% k_rho = kb * abs(rho - rho_s); % Distance from source to each grid point
% u_inc = -1j * (1/4) * besselh(0, 2, k_rho); % Incident field
% 
% % Extract real and imaginary parts
% u_real = real(u_inc);
% u_imag = imag(u_inc);
% u_abs = abs(u_inc);
% 
% % Plotting
% figure;
% 
% % Real part
% subplot(1, 3, 1);
% imagesc(x, y, u_real);
% %title('Real Part');
% xlabel('x');
% ylabel('y');
% colorbar;
% axis equal tight;% % 合并标题和数学符号
% title(['Real Part of $$\hat{u}_{inc}$$'], 'Interpreter', 'latex', 'FontSize', 14, 'HorizontalAlignment', 'center');
% 
% % Imaginary part
% subplot(1, 3, 2);
% imagesc(x, y, u_imag);
% %title('Imaginary Part of $$\hat{u}_{inc}$$''Interpreter', 'tex');
% xlabel('x');
% ylabel('y');
% colorbar;
% axis equal tight;
% title(['Imaginary Part of $$\hat{u}_{inc}$$'], 'Interpreter', 'latex', 'FontSize', 14, 'HorizontalAlignment', 'center');
% 
% % Absolute value
% subplot(1, 3, 3);
% imagesc(x, y, u_abs);
% xlabel('x');
% ylabel('y');
% colorbar;
% axis equal tight;
% title(['Absolute Value of $$\hat{u}_{inc}$$'], 'Interpreter', 'latex', 'FontSize', 14, 'HorizontalAlignment', 'center');
% % Ensure proper orientation
% set(gca, 'YDir', 'reverse');
% 
% % Adjust layout
% sgtitle('Incident Field Visualization');
% 
% % Display the plot
% colormap('jet'); % Choose colormap
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5555555
% % Plotting Configuration Sketch
% figure;
% hold on;
% 
% % Define object domain D
% lambda = 2 * pi / kb; % Wavelength
% x_limit = lambda; % Domain limits
% y_limit = lambda;
% 
% % Object domain boundaries
% rectangle('Position', [0, 0, lambda, lambda], 'EdgeColor', 'k');
% 
% % Source position
% plot(real(rho_s), imag(rho_s), 'ro', 'MarkerFaceColor', 'r'); % Source position
% text(real(rho_s), imag(rho_s), ' Source Position', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right'); % Source position annotation
% 
% hold off;
% 
% title('Configuration Sketch');
% xlabel('x');
% ylabel('y');
% axis equal;
% %% contrast function
% % Compute contrast function within Dobj
% % Assume cb = 1 for background wave speed
% cb = 1;
% c_r = sqrt(X.^2 + Y.^2); % Distance from the origin for each grid point
% chi = zeros(size(X));
% inDobj = (X >= Dobj(1) & X <= Dobj(1) + Dobj(3) & Y >= Dobj(2) & Y <= Dobj(2) + Dobj(4));
% chi(inDobj) = 1 - (cb ./ c_r(inDobj)).^2;
% 
% % Plotting the contrast function
% figure;
% imagesc(x, y, chi);
% colormap('jet'); % Choose colormap
% colorbar; % Add colorbar
% axis equal tight;
% rectangle('Position', Dobj, 'EdgeColor', 'k'); % Plot the object domain rectangle
% title('Contrast Function \chi(\rho)');
% xlabel('x');
% ylabel('y');
%% Reciver domain
% Define receiver domain Drec
L = 3 * lambda; % Length of the line segment
y_rec = 1.5 * lambda; % y-coordinate for the line segment

% Endpoints of the line segment
endpoint1 = [-lambda, y_rec];
endpoint2 = [2 * lambda, y_rec];

% Plotting Configuration Sketch
figure;
hold on;

% Object domain boundaries
rectangle('Position', [0, 0, lambda, lambda], 'EdgeColor', 'k');
% Adding a patch for the object domain to be included in the legend
patch([0 lambda lambda 0], [0 0 lambda lambda], 'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none');

% Source position
xs = lambda / 2;
ys = 10 * lambda;
plot(xs, ys, 'ro', 'MarkerFaceColor', 'r');
text(xs, ys, ' Source Position', 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');

% Receiver domain Drec
line([endpoint1(1), endpoint2(1)], [endpoint1(2), endpoint2(2)], 'Color', 'b', 'LineWidth', 2);
text(endpoint1(1), endpoint1(2), sprintf(' (%.1f, %.1f)', endpoint1(1), endpoint1(2)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
text(endpoint2(1), endpoint2(2), sprintf(' (%.1f, %.1f)', endpoint2(1), endpoint2(2)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'left');

% Set axis limits and ticks
xlim([-lambda, 2*lambda]);
ylim([0, ys + lambda]);

% Ensure equal aspect ratio
axis equal;

% Set x and y axis ticks to intervals of 10
xticks = -60:10:60;
yticks = 0:10:60;


% Add grid lines for better visualization
grid on;
hold off;

title('Configuration Sketch');
xlabel('x');
ylabel('y');
legend('Object Domain D', 'Source Position','Receiver Domain','Location', 'Best');
% Ensure proper orientation
set(gca, 'YDir', 'reverse');
%% % contrast function
n = length(x) ;
contrast_matrix = zeros(n, n);
%Mine
R=6;
for x = 1:n
    for y = 1:n
        contrast_matrix(y, x) = contrast(x_inc_mid(x), y_inc_mid(y), 0.5*lambda, 0.5*lambda, R);
    end
end

figure()
imagesc(x_inc_mid, y_inc_mid, contrast_matrix);

axis equal tight;
colorbar;
colormap('jet');
title('Contrast Function');

%Mas
% k_rho = kb*ones(20,20);
% 
% cross
% k_rho(10:11,5:16) = 2.5*ones(2,12);
% k_rho(5:16,10:11) = 2.5*ones(12,2);
% 
% 
% contrast_matrix = (k_rho/kb).^2-1;
% 
% figure;
% imagesc(contrast_matrix)
% colorbar;
% axis tight



%% % Define number of grid points on Drec
M = 100;
x_rec_grid = -lambda:3*lambda/(M-1):2*lambda;
y_rec = 1.5*lambda;

% Reshape contrast function
contrast_vec = reshape(contrast_matrix, n*n, 1); %x
% 
% % Build the system matrix A
% a = h/2;
% u_inc_diag = diag(reshape(u_inc, n*n, 1));
% G = gen_G(a, kb, M, n, x_rec_grid, y_rec, h);
% [u_sc, u_sc_broaden, A, A_broaden] = gen_system(G, u_inc_diag, contrast_vec);

%Build A
% % 
% A = zeros(M, 20 * 20);
% index = 1;  % 初始化索引A
% for i = 1:m
%     for j = 1:n
%         grid_x = X(i, j);
%         grid_y = Y(i, j);
%         a = gen_A(kb, grid_x, grid_y ,xs, ys, x_rec_grid, y_rec, h);
%      
%         A(:, index) = a;
%         index = index + 1;
%         % print()
%     end
% end
x_rec_grid2 = x_rec_grid';
y_rec_vector = repmat(y_rec, M, 1);  
A = gen_A(kb,X,Y,xs,ys,x_rec_grid2,y_rec_vector,h);



% Singular value decomposition
S_A = svd(A);
[U, S, V] = svd(A);
figure()
scatter(1:length(S_A), S_A);
title(['M = ', num2str(M)]);
%reconstruct
u_sc = A*contrast_vec; %u_sc

%pin
contrast_vec_hat = pinv(A)*u_sc;
contrast_matrix_hat = reshape(contrast_vec_hat, n, n);
%pin%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% contrast_vec_hat = pinv(A_broaden)*u_sc_broaden;
% contrast_matrix_hat = reshape(contrast_vec_hat, n, n);
%%%%%%%%%%%%%%%%%%%%%%
% %svd
% [U1, S1, V1] = svd(A_broaden);
% D_dagger = zeros(size(A_broaden'));
% D_dagger(1:min(size(A_broaden)), 1:min(size(A_broaden))) = diag(1 ./ diag(S1));
% %D_dagger = diag(1 ./ diag(S));  % 
% contrast_vec_hat = V1 * D_dagger * U1' * u_sc_broaden;  
% contrast_matrix_hat = reshape(contrast_vec_hat, n, n);
% % contrast_vec_hat = pinv(A)*u_sc_broaden;
% % contrast_matrix_hat = reshape(contrast_vec_hat, n, n);
%%%%%%%%%%%%%%%%%%%%%%
% %svd
% 
% D_dagger = zeros(size(A'));
% D_dagger(1:min(size(A)), 1:min(size(A))) = diag(1 ./ diag(S));
% %D_dagger = diag(1 ./ diag(S));  % 
% 
% contrast_vec_hat = V* D_dagger * U' * u_sc;  
% contrast_matrix_hat = reshape(contrast_vec_hat, n, n);

figure()
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat));
axis equal tight;
colorbar;
colormap('jet');
title('Reconstructed Contrast Function-M= ',num2str(M));

%% ADD NOISE
% Desired signal-to-noise ratio (SNR) in dB
desired_snr_db = 20;

% Calculate the noise level based on the desired SNR
% SNR = 10 * log10(signal_power / noise_power)
% 10^(desired_snr_db / 10) = signal_power / noise_power
% noise_power = signal_power / 10^(desired_snr_db / 10)
signal_power = norm(u_sc)^2 / length(u_sc);  % Assuming unit variance for u_sc
noise_power = signal_power / 10^(desired_snr_db / 10);
noise_level = sqrt(noise_power);  % Standard deviation of noise

% Generate noise vector
noise_vector = noise_level * randn(size(u_sc));

% Add noise to u_sc
u_sc_noisy = u_sc + noise_vector;

% Reconstruct contrast function with noisy data vector
contrast_vec_hat_noisy = pinv(A) * u_sc_noisy;
contrast_matrix_hat_noisy = reshape(contrast_vec_hat_noisy, n, n);

% Plot noisy reconstruction result
figure();
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat_noisy));
axis equal tight;
colorbar;
colormap('jet');
title(sprintf('Reconstructed Contrast Function with Noise(SNR %.2fdB)', 10 * log10(signal_power / noise_level^2)));

% Show comparison between original contrast function and noisy reconstruction
figure();
subplot(1, 2, 1);
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat));
axis equal tight;
colorbar;
colormap('jet');
title('No Noise Reconstruction');
subplot(1, 2, 2);
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat_noisy));
axis equal tight;
colorbar;
colormap('jet');
title(sprintf('Noisy Reconstruction (SNR %.2f dB)', 10 * log10(signal_power / noise_level^2)));
%% ADD noise while seperate the real and imaginary
% Desired signal-to-noise ratio (SNR) in dB
desired_snr_db = 20;

% Calculate the noise level based on the desired SNR
% SNR = 10 * log10(signal_power / noise_power)
% 10^(desired_snr_db / 10) = signal_power / noise_power
% noise_power = signal_power / 10^(desired_snr_db / 10)
signal_power = norm(u_sc_broaden)^2 / length(u_sc_broaden);  % Assuming unit variance for u_sc
noise_power = signal_power / 10^(desired_snr_db / 10);
noise_level = sqrt(noise_power);  % Standard deviation of noise

% Generate noise vector
noise_vector = noise_level * randn(size(u_sc_broaden));
% Add noise to u_sc
u_sc_noisy_broaden = u_sc_broaden + noise_vector;

% 用带有噪声的数据向量进行对比度函数的重构
contrast_vec_hat_noisy_broaden = pinv(A_broaden) * u_sc_noisy_broaden;
contrast_matrix_hat_noisy_broaden = reshape(contrast_vec_hat_noisy, n, n);

% 
figure();
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat_noisy_broaden));
axis equal tight;
colorbar;
colormap('jet');
title('reconstruction without noise');

% 
figure();
subplot(1, 2, 1);
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat));
axis equal tight;
colorbar;
colormap('jet');
title('reconstruction without noise');
subplot(1, 2, 2);
imagesc(x_inc_mid, y_inc_mid, abs(contrast_matrix_hat_noisy));
axis equal tight;
colorbar;
colormap('jet');
title(sprintf('reconstruction with noise (noise level %.2f)', 10 * log10(signal_power / noise_level^2)));
