function [ figure_handle, listbox_handle1, listbox_handle3, instructions_handle] = makescreen( class2pick1)
%function [ figure_handle, listbox_handle1, listbox_handle3, instructions_handle] = makescreen( class2pick1, MCconfig )
%For Imaging FlowCytobot roi viewing; Use with manual_classify scripts;
%Sets up a graph window for manual identification from a roi collage (use
%fillscreen.m to add the rois);
%Heidi M. Sosik, Woods Hole Oceanographic Institution, 30 May 2009

%INPUT:
%class2pick1 - cell array of class labels
%MCconfig - configuration structure from get_MCconfig.m
%
%OUTPUT:
%figure_handle - handle to figure window
%listbox_handle1 - handle to category list box on left
%listbox_handle3 - handle to category list box on right for long main list
%instructions_handles - handle to text box for instructions
%
%Sept 2014, revised for more robust handling of screen size issues
%April 2015, revised to remove subdivide functionality and recast for manual_classify_4_1

global category MCconfig MCflags new_classcount new_filecount filelist filecount

screen = get(0, 'ScreenSize');

figure_handle = figure;
set(figure_handle, 'outerposition', screen, 'color', [1 1 1])
set(figure_handle, 'units', 'inches')
tpos = get(figure_handle, 'position');
lwdth = .8/tpos(4); %.8 inches as fraction of screen, listbox width
lmargin = .3/tpos(4); %.2 inches as fraction of screen, bottom margin below list boxes
instructions_handle = NaN;

if ~isempty(class2pick1), %edited 1/12/10 to fix typo pick2 --> pick1
    
    switch MCconfig.alphabetize
        case 'yes'
            [~, ix] = sort(lower(class2pick1));%sorting class2pick1
            numstr_to_display = (1:length(class2pick1))';%sorting the indexes class2pick1
            str = cellstr([num2str(numstr_to_display(ix), '%03d') repmat(' ',length(class2pick1),1) char(class2pick1{ix})]);
        case 'no'
            str = cellstr([num2str((1:length(class2pick1))', '%03d') repmat(' ',length(class2pick1),1) char(class2pick1)]);
        otherwise
            warning('You should choose ''yes'' or ''no'' for the variable MCconfig.alphabetize_list. The list will not be alphabetized for now')
            str = cellstr([num2str((1:length(class2pick1))', '%03d') repmat(' ',length(class2pick1),1) char(class2pick1)]);
    end
    str1 = str; 
    listbox_handle3 = NaN;
    if length(str) > MCconfig.maxlist1,
        str1 = str(1:MCconfig.maxlist1);
        str2 = str(MCconfig.maxlist1+1:end);
        listbox_handle3 = uicontrol('style', 'listbox', 'string', str2, 'ForegroundColor', 'r', 'callback', @select_category_callback);
        set(listbox_handle3, 'units', 'normalized', 'position',[1-lwdth lmargin lwdth 1-lmargin])
    end;
    listbox_handle1 = uicontrol('style', 'listbox', 'string', str1, 'ForegroundColor', 'r', 'callback', @select_category_callback);
    set(listbox_handle1, 'units', 'normalized', 'position', [0 lmargin lwdth 1-lmargin]);
    instructions_handle = uicontrol('style', 'text');
    set(instructions_handle, 'units', 'normalized', 'position', [lwdth*3 lmargin lwdth*4 lmargin]);% tpos)
    set(instructions_handle, 'string', ['Use mouse button to choose category. Then click on ROIs. Hit ENTER key to stop choosing.'])    
end;

set(figure_handle, 'menubar', 'none')
%step_flag  = 0;
%file_jump_flag = 0;
change_menu_handle =  uimenu(figure_handle, 'Label', 'Change &Class');
next_menu_handle =  uimenu(change_menu_handle, 'Label', '&Next Class', 'callback', {@class_step_amount_callback, 1}, 'Accelerator', 'n');
prev_menu_handle =  uimenu(change_menu_handle, 'Label', '&Previous Class', 'callback', {@class_step_amount_callback, -1}, 'Accelerator', 'p');
jump_menu_handle = uimenu(change_menu_handle, 'Label', '&Jump to Selected Class', 'callback', @jump_class_callback, 'Accelerator', 'j');
file_change_menu_handle =  uimenu(figure_handle, 'Label', 'Change &File');
file_next_menu_handle =  uimenu(file_change_menu_handle, 'Label', '&Next File', 'callback', {@jump_file_callback, 1}, 'Accelerator', 'l');
file_prev_menu_handle =  uimenu(file_change_menu_handle, 'Label', '&Previous File', 'callback', {@jump_file_callback, -1}, 'Accelerator', 'k');
file_jump_menu_handle = uimenu(file_change_menu_handle, 'Label', '&Jump to Selected File', 'callback', {@jump_file_callback, 0}, 'Accelerator', 'm');
configure_menu_handle = uimenu(figure_handle, 'Label', '&Options', 'callback', @change_config_callback);
quit_menu_handle =  uimenu(figure_handle, 'Label', '&Quit');
quit_script_menu_handle =  uimenu(quit_menu_handle, 'Label', '&Quit manual_classify', 'callback', @stopMC_callback, 'Accelerator', 'q');
exit_menu_handle =  uimenu(quit_menu_handle, 'Label', 'E&xit MATLAB', 'callback', 'exit', 'Accelerator', 'x');
u = uicontrol(gcf, 'style', 'radiobutton', 'units', 'normalized');
set(u, 'position', [lwdth*1.1 lmargin*1.5 lwdth*1.25 lmargin], 'string', 'SELECT remaining in class', 'callback', @select_remaining_callback, 'fontsize', 10)

function select_category_callback( hOBj, eventdata )
%Sets up uicontrol for picking categories; callback for class listboxes
%April 2015, revised to remove subdivide functionality and recast for
%manual_classify_4_1, converted to nested function in makescreen
    if gco == listbox_handle1
        MCflags.button = 1;
        h = listbox_handle1;
    elseif gco == listbox_handle3,
        MCflags.button = 3;
        h = listbox_handle3;
    else
        MCflags.button = NaN;
    end;
    str = get(h, 'string');
    category = char(str(get(h, 'value')));
    set(instructions_handle, 'string', ['Click on ' category...
        ' images; then ENTER key to save results before changing categories. ENTER key for new page.'], 'foregroundcolor', 'k')
    %xl = xlim; yl = ylim;
    %h = fill([xl([1,2,2,1])]', yl([1,1,2,2])', 'w', 'facealpha', .7);
    ReleaseFocus(gcf)
    robot_pressESC(1)
    %delete(h)
end

function class_step_amount_callback( hOBj, eventdata, amount )
% callback function for 'next class' and 'previous class' menu options in
% manual_classify for IFCB
    MCflags.class_step = amount;
    robot_pressCR(1)
end

function jump_class_callback( hOBj, eventdata )
%function [  ] = class_change_amount( hOBj, eventdata, direction )
% callback function for 'jump to selected class' menu option in
% manual_classify for IFCB
% Heidi M. Sosik, Woods Hole Oceanographic Institution, March 2015
    MCflags.class_jump = 1;
    new_classcount = str2num(category(1:3));
    robot_pressCR(1) % one carriage return
end

function jump_file_callback( hOBj, eventdata, jump_type )
% callback function for 'jump to selected file' menu option in
% manual_classify for IFCB
    MCflags.file_jump = 1;
    if jump_type == 0, %case for jump to selected file
        [new_filecount,v] = listdlg('PromptString','Select a file:', 'SelectionMode','single','ListString',filelist, 'ListSize', [300 400], 'initialvalue', filecount);
        if v == 0 %user cancelled
            new_filecount = NaN;
            MCflags.file_jump = 0;
        end;
    else %case for step forwared or backward
        new_filecount = filecount + jump_type;
        if new_filecount > length(filelist)
            set(instructions_handle, 'string', ['LAST FILE! Use ''Quit'' menu to stop classifying.'], 'foregroundcolor', 'r', 'fontsize', 16)
            new_filecount = filecount;
        end
    end
    robot_pressCR(1)
end

function change_config_callback( hOBj, eventdata )
%   callback function for Options menu in manual_classify
    prompt = {'Set size for image display' 'Image resizing factor (1 = none)'};
    defaultanswer={num2str(MCconfig.setsize),num2str(MCconfig.imresize_factor)};
    user_input = inputdlg(prompt,'Configure', 1, defaultanswer);
    if ~isempty(user_input)
        [val status] = str2num(user_input{1});
        if status && rem(val,1) ~= 0, status = 0; end
        while ~status
            uiwait(msgbox(['Set size must be an integer']))
            user_input(1) = defaultanswer(1);
            user_input = inputdlg(prompt,'Configure', 1, user_input);
            [val status] = str2num(user_input{1});
            if status && rem(val,1) ~= 0, status = 0; end
        end
        MCconfig.setsize = str2num(user_input{1});
        [val status] = str2num(user_input{2});
        while ~status
            uiwait(msgbox(['Resize factor must be a number']))
            user_input(2) = defaultanswer(2);
            user_input = inputdlg(prompt,'Configure', 1, user_input);
            [val status] = str2num(user_input{2});
        end
        MCconfig.imresize_factor = str2num(user_input{2});
    end
    %ReleaseFocus(gcf)
    %drawnow
end

function stopMC_callback( hOBj, eventdata )
%   callback for quit from manual_classify menu entry
    MCflags.file_jump = 1;
    new_filecount = length(filelist)+1;
    robot_pressCR(1)
end

function select_remaining_callback( hOBj, eventdata )
%   callback function for 'select remaining in class' radio button in manual_classify
    MCflags.select_remaining = 1;
    ReleaseFocus(gcf)
    robot_pressCR(1)
    set(hOBj, 'value',0) %unselect the button
end



end

