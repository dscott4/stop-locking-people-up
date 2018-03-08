dataset=xlsread('crimeNeverPays.csv');
features=dataset(:,1:10);
Reoffension=dataset(:,11);
ind=find(Reoffension==1);
numdaysfeatures=features(ind,:);
numdays=dataset(ind,12);
mdl=fitcsvm(features,Reoffension,'KernelFunction','RBF');
ypred=predict(mdl,features);
thing=(ypred~=Reoffension);
error=sum(thing)/length(thing);


features2=features;
features2(:,3:4)=[];
mdl2=fitcsvm(features2,Reoffension,'KernelFunction','RBF');
ypred2=predict(mdl2,features2);
thing2=(ypred2~=Reoffension);
error2=sum(thing2)/length(thing2);