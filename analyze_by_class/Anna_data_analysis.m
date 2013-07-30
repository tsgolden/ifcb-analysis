%code by Emily B for Anna Nisi to use ciliate count result files 6/28/13

load \\raspberry\d_work\IFCB1\ifcb_data_mvco_jun06\Manual_fromClass\summary\count_manual_10Jun2013_day.mat  %loads your result file for counts. path and date may have to be adjusted

%load \\raspberry\d_work\IFCB1\ifcb_data_mvco_jun06\Manual_fromClass\summary\count_biovol_size_manual_17May2013.mat %loads your result file for biovolume, size, etc

%code for using count data

[ ind_ciliate, class_label ] = get_ciliate_ind( class2use, class2use ); %the get_ciliate_ind function takes the 'class2use' variable and 
%finds the ciliate categories.  It gives you 'class_label' which are the
%names for all categories and 'ind_ciliate' which gives you the indexes for
%finding and useing the ciliate categories
ciliate_classcount=(classcount_bin(:,ind_ciliate)./ml_analyzed_mat_bin(:,ind_ciliate)); %by using the 'ind_ciliate' variable, you pull out
%ciliate data when computing the count per ml.
ciliate_classcount(isnan(ciliate_classcount))=0; %replaces the NaNs with zeros

ciliate_label=class_label(ind_ciliate);
for label=1:length(ciliate_label)
    Ciliate_abundance_Struct.(ciliate_label{label})=ciliate_classcount(:,label);
end

    
classes = fields(Ciliate_abundance_Struct); %goes through all elements of structure and makes two new structures (day and week binned matrices)
for classcount=1:length(classes),
    [mdate_mat, Ciliate_day_mat.(classes{classcount}), yearlist, yd ] = timeseries2ydmat( matdate_bin, Ciliate_abundance_Struct.(classes{classcount}));
    [Ciliate_week_mat.(classes{classcount}), mdate_wkmat, yd_wk] =ydmat2weeklymat(Ciliate_day_mat.(classes{classcount}), yearlist);
    [Ciliate_fortnight_mat.(classes{classcount}), mdate_fnmat, yd_fn] =ydmat2fortnightmat(Ciliate_day_mat.(classes{classcount}), yearlist);
end

 for classcount=1:length(classes); %plots week bins of all ciliate data
     figure
     h=plot(yd_wk, Ciliate_week_mat.(classes{classcount}), '.-'); %h=plot(yd_wk, Ciliate_week_mat.Ciliate_mix, '.-');
     legend(num2str((2006:2013)'), 'location', 'south');
     title(classes{classcount}, 'fontsize', 10, 'fontname', 'arial'); %title('Ciliate_mix', 'fontsize', 10, 'fontname', 'arial');
     ylabel('Abundance (cell mL^{-1} \mum{-1})', 'fontsize', 12);
     set(h(8),'DisplayName','2013','Color',[0 0 0]);
     datetick('x', 3, 'keeplimits');
     set(gca,'xgrid','on');
 end 
 
 %Ciliate_mix_mat=Ciliate_mat.Ciliate_mix;%use this if you want to bring a certain element out of the structures
 %Ciliate_mix_mat(isnan(Ciliate_mix_mat))=0; %once you take an element out of the structre, you can convert the NaNs to zeros for nice figures and such
 %Ciliate_mix_2006=Ciliate_mix_mat(:,1); %if you just want to look at a
 %certain year
  %how to make a vector out of all the columns in a structure (for LSA):
 %Mesodinium_sp = Ciliate_week_mat.Mesodinium_sp(:)

