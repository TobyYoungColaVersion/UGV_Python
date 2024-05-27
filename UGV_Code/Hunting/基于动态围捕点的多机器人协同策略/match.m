function matchs=match(D)
d_max_i=max(max(D))*ones(1,4);
d_max_j=d_max_i';
for i=1:4
    [temp_i,temp_j]=find(D==min(min(D)));
    matchs(i,:)=[temp_i(1),temp_j(1)];
    D(matchs(i,1),:)=d_max_i;]
    D(:,matchs(i,2))=d_max_j;
end
