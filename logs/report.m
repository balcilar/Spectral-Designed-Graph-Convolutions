% clear all

%a=xlsread('citeseer_result.xlsx');

%a=xlsread('pubmed_result.xlsx');

%a=xlsread('cora_result.xlsx');

clear s

for i=1:20
    I=find(a(:,1)==i-1);
    a1=a(I,:);    
  
    
    [u y]=max(a1(:,11));
    yy=find(a1(:,11)==u);
    y=yy(1);
    s(i,1)=a1(y,13);

    [u y]=min(a1(:,9));    
    s(i,2)=a1(y,13);
    
    
end
format long
mean(s)
std(s)
    


