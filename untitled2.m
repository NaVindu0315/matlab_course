% Data Statistics

% Load the dataset
load fisheriris.mat;

% Report the total number of rows
totalRows = size(meas, 1);
fprintf('Total number of rows (objects, cases): %d\n', totalRows);

% For each column (feature) from 1 to 4
for col = 1:4
    columnData = meas(:, col); % Extract the column data
    
    fprintf('Statistics for Column %d:\n', col);
    fprintf('  Mean: %f\n', mean(columnData));
    fprintf('  Standard Deviation: %f\n', std(columnData));
    fprintf('  Maximum: %f\n', max(columnData));
    fprintf('  Minimum: %f\n', min(columnData));
    fprintf('  Root Mean Square: %f\n\n', rms(columnData));
end
