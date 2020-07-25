format compact
clear
clc

%% Load data - Split data
data=load('../Datasets/superconduct.csv');
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);
Perf=zeros(1,4);

%% Evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% Keep only the number of features we want and not all of them
[ranks, weights] = relieff(data(:,1:end-1), data(:, end), 100);

%% FINAL TSK MODEL
fprintf('\n *** TSK Model with 15 features and radii 0.3 - Substractive Clustering\n');

f = 15;
radii = 0.3;

training_data_x = trnData(:,ranks(1:f));
training_data_y = trnData(:,end);

validation_data_x = chkData(:,ranks(1:f));
validation_data_y = chkData(:,end);

test_data_x = tstData(:,ranks(1:f));
test_data_y = tstData(:,end);%% TRAIN TSK MODEL

%% MODEL WITH 15 FEATURES AND 4 RULES

% Generate the FIS
fprintf('\n *** Generating the FIS\n');

% As input data I give the train_id's that came up with the
% partitioning and only the most important features
% As output data is just the last column of the test_data that
% are left
init_fis = genfis2(training_data_x, training_data_y, radii);
rules = length(init_fis.rule);
% Plot some input membership functions
figure;

for i = 1 : f
    [x, mf] = plotmf(init_fis, 'input', i);
    plot(x,mf);
    hold on;
end

suptitle('TSK model : some membership functions before training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/mf_before_training.png');

% Tune the fis
fprintf('\n *** Tuning the FIS\n');

% Set some options
% The fis structure already exists
% set the validation data to avoid overfitting
anfis_opt = anfisOptions('InitialFIS', init_fis, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);

[trn_fis, trainError, stepSize, valFis, valError] = anfis([training_data_x training_data_y], anfis_opt);

% Evaluate the fis
fprintf('\n *** Evaluating the FIS\n');

% No need to specify specific options for this, keep the defaults
Y = evalfis(test_data_x, valFis);

%% METRICS
error = Y - test_data_y;

R2=Rsq(Y,test_data_y);
MSE=mse(Y,test_data_y);
RMSE=sqrt(MSE);
NMSE=(sum((test_data_y - Y) .^ 2) / length(Y)) / var(test_data_y);
NDEI=sqrt(NMSE);

Perf(1,:)=[R2;RMSE;NMSE;NDEI];

% Plot the metrics
figure;
plot(1:length(test_data_y), test_data_y, '*r', 1:length(test_data_y), Y, '.b');
title('Output');
legend('Reference Outputs', 'Model Outputs');
saveas(gcf, 'Final_TSK_model/output.png')

figure;
plot(error);
title('Prediction Errors');
saveas(gcf, 'Final_TSK_model/error.png')

figure;
plot(1:length(trainError), trainError, 1:length(valError), valError);
title('Learning Curve');
legend('Traning Set', 'Test Set');
saveas(gcf, 'Final_TSK_model/learningcurves.png')

% Plot the input membership functions after training
figure;

for i = 1 : f
    [x, mf] = plotmf(valFis, 'input', i);
    plot(x,mf);
    hold on;
end

suptitle('Final TSK model : some membership functions after training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/mf_after_training.png');

fprintf('MSE = %f RMSE = %f R^2 = %f NMSE = %f NDEI = %f\n', MSE, RMSE, R2, NMSE, NDEI)







