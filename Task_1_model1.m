format compact
clear
clc

%% Load data - Split data
data=load('../Datasets/airfoil_self_noise.dat');
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);
Perf=zeros(1,4);

%% Evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);



Model1 = genfis1(trnData, 2, 'gbellmf','constant');


%{
Model2 = genfis1(trnData, 3, 'gbellmf','constant');
Model3 = genfis1(trnData, 2, 'gbellmf','linear');
Model4 = genfis1(trnData, 3, 'gbellmf','linear'); 
%}

[trnFis,trnError,~,valFis,valError]=anfis(trnData,Model1,[100 0 0.01 0.9 1.1],[],chkData);




%% Validation
figure(1);
plot([trnError valError],'LineWidth',2); grid on;
xlabel('# of Iterations'); ylabel('Error');
legend('Training Error','Validation Error');
title('ANFIS Hybrid Training - Validation');
Y=evalfis(chkData(:,1:end-1),valFis);

% metrics
R2=Rsq(Y,chkData(:,end));
RMSE=sqrt(mse(Y,chkData(:,end)));
NMSE=(sum((chkData(:,end) - Y) .^ 2) / length(Y)) / var(chkData(:,end));
NDEI=sqrt(NMSE);


Perf(1,:)=[R2;RMSE;NMSE;NDEI];
figure(2);



%% Results Table
varnames={'Rsquared','RMSE','NMSE','NDEI'};
Perf=array2table(Perf,'VariableNames',varnames);

%~ Membership Functions of Initial Model ~%
plotMFs(Model1,size(trnData,2)-1);
suptitle('TSK model : some membership functions before training');
%~ Membership Functions of Trained Model ~%
figure(3);
plotMFs(valFis,size(trnData,2)-1);
suptitle('TSK model : some membership functions after training');


