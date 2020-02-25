clear s
for i=0:19 
   a=csvread(['enzyms_fullfeat_custom_nk4_' num2str(i) '.csv']);
    a(1,:)=[];a(:,1)=[];
    [r t]=max(sum(a'));
    s(i+1,:)=a(t,:);
end
mean(mean(s'/60))
mean(std(s'/60))
std(mean(s'/60))