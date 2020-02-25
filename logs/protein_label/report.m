clear s
for i=0:10
   a=csvread(['protein_label_custom_nk2_' num2str(i) '.csv']);
    a(1,:)=[];a(:,1)=[];
    [r t]=max(sum(a'));
    s(i+1,:)=a(t,:);
end
mean(sum(s')/1113)
std(sum(s')/1113)