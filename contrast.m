
function contrast_value = contrast(x, y, x_center, y_center, R)


% Calculate distance from circle edge
distance_from_edge = R - sqrt((x - x_center).^2 + (y - y_center).^2);

% Normalize distance to [0, 1]
contrast_value = distance_from_edge / R;

% Clip values outside [0, 1] (optional)
contrast_value = max(0, min(1, contrast_value));

end