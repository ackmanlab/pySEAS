import numpy as np
import os
import matplotlib
import tkinter as tk
import tkinter.messagebox

# Set the matplotlib backend to tk figures.
# If imported, cannot plot to regular matplotlib figures!
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from seas.signalanalysis import lag_n_autocorr, sort_noise, short_time_fourier_transform
from seas.waveletAnalysis import waveletAnalysis
from seas.hdf5manager import hdf5manager
from seas.defaults import config
from seas.ica import rebuild_eigenbrain
from seas.domains import get_domain_map, domain_map, get_domain_edges


def run_gui(components, rotate=0, savepath=None, default_assignment=None):
    '''
    Create a tkinter GUI to select noise components from ica-decomposition components file.

    Returns toggle, a boolean array 
    of either True or False.  Components that have been determined to be 
    noise are 'True', components to keep are 'False'.
    Optional toggle input is a boolean of starting values 
    for noise_components.
    '''

    print('\nStarting ICA Component Selection GUI\n-----------------------')
    print(
        'If you experience problems with this GUI, check your tk version with the following terminal command:'
    )
    print('\tpython -m tkinter')
    print(
        'This will launch a GUI that tells you your tk version.  Make sure it is >=8.6'
    )
    print(
        "If installed using conda, use 'conda install -c anaconda tk' to update"
    )

    if type(components) is str:
        f = hdf5manager(components)
        print('loading components...')
        components = f.load()
        load_hdf5 = True
    else:
        load_hdf5 = False

    if savepath is not None:
        f = hdf5manager(savepath)
        load_hdf5 = True

    assert type(components) is dict, 'Components were not in expected format'

    # load all components from dict
    eig_vec = components['eig_vec']
    if 'thresholds' in components.keys():
        thresholds = components['thresholds']
    roimask = components['roimask']
    shape = components['shape']
    t, x, y = shape
    eigb_shape = (x, y)

    # Find number of components
    n_components = eig_vec.shape[1]
    print('number of components:', n_components)

    # start timecourses variable for storing rebuilt timecourses of PCs
    if 'timecourses' in components:
        print('timecourses found')
        timecourses = components['timecourses']
    else:
        print('timecourses not found in components')
        print('Initializing empty timecourse vector')
        timecourses = components['timecourses']

    # start noise_components variable for listing which components are noise
    if ('noise_components' in components) and ('cutoff' in components):
        noise_components = components['noise_components']
        cutoff = components['cutoff']
    else:
        print('calculating noise components...')
        noise_components, cutoff = sort_noise(timecourses)
    maxval = np.argmax(noise_components == 1)

    if 'lag1_full' in components:
        lag1 = components['lag1_full']
        _, _, log_pdf = sort_noise(timecourses, lag1=lag1, return_logpdf=True)
    else:
        lag1 = lag_n_autocorr(timecourses, 1, verbose=False)
        _, _, log_pdf = sort_noise(timecourses, return_logpdf=True)

    # start toggle variable for checking which components shouldn't
    # be included
    if 'artifact_components' in components:
        toggle = components['artifact_components']
    else:
        print('initializing artifact_components toggle')
        toggle = np.zeros((n_components,), dtype='uint8')

    if 'flipped' in components:
        flipped = components['flipped']

        timecourses = timecourses * flipped[:, None]
        eig_vec = eig_vec * flipped

    if 'domain_ROIs' in components:
        domain_ROIs = components['domain_ROIs']
        if rotate > 0:
            domain_ROIs = np.rot90(domain_ROIs, rotate)

        if 'region_assignment' in components:
            region_assignment = components['region_assignment']
        else:
            print('initializing region_assignment vector')
            n_domains = int(np.nanmax(domain_ROIs) + 1)
            region_assignment = np.zeros((n_domains,))

            if default_assignment is not None:
                try:
                    region_assignment += default_assignment
                except:
                    raise TypeError('Region Assignment was invalid {0}'.format(
                        default_assignment))
            else:
                region_assignment[:] = np.nan
    else:
        print('domain_ROIs not found')
        domain_ROIs = None

    # Load mask indexing for faster rebuilding of eigenbrains
    if roimask is not None:
        maskind = np.where(roimask.flat == 1)
    else:
        maskind = None

    region_cm = config['colormap']['domains']
    component_cmap = config['colormap']['components']
    corr_cmap = config['colormap']['correlation']

    regions = config['regions']

    # convert from defaults dict to sorted list of tuples
    keylist = []
    valuelist = []

    for key in regions:
        # print(key, regions[key])
        keylist.append(key)
        valuelist.append(regions[key])

    sortindex = [i[0] for i in sorted(enumerate(valuelist), key=lambda x: x[1])]

    regions = []
    for i in sortindex:
        regions.append((keylist[i], valuelist[i]))

    # general font settings
    LARGE_FONT = ('Verdana', 12)

    togglesave = [True]  # default on for easy click-to-save figures
    toggledebug = [False]  # default off -- don't load ICA pixel properties

    def saveFigure(fig_handle):
        print('figure was pressed!')
        # callback for clicks on image.
        # Change color and toggle value

        if togglesave[0]:
            print('trying to save..')
            file_path = tk.filedialog.asksaveasfilename()
            print(file_path)

            if type(file_path) is str:  # If path was provided
                # If there was no extension, add .png
                if os.path.splitext(file_path)[1] == '':
                    file_path += '.png'

                if os.path.isfile(file_path):
                    print('file already exists!')
                    yn = tk.messagebox.askquestion(
                        'Overwriting File',
                        'The following file already exists, would you '
                        'like to overwrite it?' + '\n' + file_path)

                    if yn == 'yes':
                        fig_handle.savefig(file_path)
                        print('File saved to:', file_path)
                    else:
                        print('Not overwriting')

                else:
                    print('file doesnt exist (yet)')
                    fig_handle.savefig(file_path)
                    print('File saved to:', file_path)
            else:
                print('No file selected')
        else:
            print('Saving functionality is turned off')

    # Create main application as tk.Tk
    class PCAgui(tk.Tk):  #<- inherits from Tk class

        def __init__(self, *args, **kwargs):
            tk.Tk.__init__(self, *args, **kwargs)  # initialize the tk class
            tk.Tk.wm_title(self, 'Component Viewer')
            # every page initializes a container to hold its content
            container = tk.Frame(self)
            container.pack(side='top', fill='both', expand=True)

            # set page expansion properties
            container.grid_rowconfigure(0, weight=1)
            # row/columns expand equally
            container.grid_columnconfigure(0, weight=1)

            # Make the menu bar (top banner)
            menubar = tk.Menu(container)

            filemenu = tk.Menu(menubar, tearoff=0)
            filemenu.add_separator()
            filemenu.add_command(label='save', command=self.quit)
            filemenu.add_command(label='exit',
                                 command=lambda: self.cancelcallback(toggle))
            menubar.add_cascade(label='file', menu=filemenu)

            def toggleFigSaving():
                togglesave[0] = not togglesave[0]

            def toggleIcaDebug():
                toggledebug[0] = not toggledebug[0]

            editmenu = tk.Menu(menubar, tearoff=0)
            editmenu.add_separator()
            editmenu.add_command(label='toggle figure saving',
                                 command=lambda: toggleFigSaving())
            editmenu.add_command(label='toggle ica pixel debug',
                                 command=lambda: toggleIcaDebug())
            menubar.add_cascade(label='edit', menu=editmenu)

            pagemenu = tk.Menu(menubar, tearoff=1)
            pagemenu.add_separator()
            pagemenu.add_command(label='view components',
                                 command=lambda: self.show_frame(PCpage))
            pagemenu.add_command(label='PC information',
                                 command=lambda: self.show_frame(PCinfo))
            pagemenu.add_command(label='Domain Correlations',
                                 command=lambda: self.show_frame(DomainROIs))
            pagemenu.add_command(label='Domain Region Assignment',
                                 command=lambda: self.show_frame(DomainRegions))
            pagemenu.add_command(label='Domain Autocorrelations',
                                 command=lambda: self.show_frame(PCautocorr))
            menubar.add_cascade(label='view', menu=pagemenu)

            tk.Tk.config(self, menu=menubar)

            # Create container to hold for all pages for page switching
            self.frames = {}

            # List all pages here:
            for F in (PCpage, PCinfo, DomainROIs, DomainRegions, PCautocorr):
                frame = F(container, self)
                self.frames[F] = frame
                # set currently active frame to StartPage
                frame.grid(row=0, column=0, sticky='nsew')

            # Initialize default page
            default_page = PCpage
            self.show_frame(default_page)

            # Global event binding commands
            self.bind("<Escape>", lambda event: self.cancelcallback(toggle))
            self.bind("s", lambda event: self.quit())
            self.bind("<F1>", lambda event: self.show_frame(PCpage))
            self.bind("<F2>", lambda event: self.show_frame(PCinfo))
            self.bind("<F3>", lambda event: self.show_frame(DomainROIs))
            self.bind("<F4>", lambda event: self.show_frame(DomainRegions))
            self.bind("<F5>", lambda event: self.show_frame(PCautocorr))

            # Use global focus to capture bindings,
            # send them to local callback manager
            self.bind("<Right>",
                      lambda event: self.current_page.callback_manager(event))
            self.bind("<Left>",
                      lambda event: self.current_page.callback_manager(event))
            self.bind("<Up>",
                      lambda event: self.current_page.callback_manager(event))
            self.bind("<Down>",
                      lambda event: self.current_page.callback_manager(event))

            # When window is closed, cancel and don't save results.
            self.protocol('WM_DELETE_WINDOW',
                          lambda: self.cancelcallback(toggle))

        # Methods for main app:
        def show_frame(self, cont):
            # selects frame, raises it
            frame = self.frames[cont]
            self.current_page = frame
            frame.tkraise()

        def cancelcallback(self, toggle):
            # quits the app, doesn't rebuild the movie
            print('Operation was cancelled.')
            toggle[0] = 100
            self.quit()

    # Create each frame and components as individual classes:
    class PCpage(tk.Frame):
        # Page to view all principal component images, toggle via clicks

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)  # parent is controller

            # Create frame for PCs, initialize PC indices
            self.nimages = 15
            self.PCpage = 0

            # initialize the title
            label = tk.Label(self,
                             text='Select Components to Remove:',
                             font=LARGE_FONT)

            # initialize navigation buttons
            navbuttons = tk.Frame(self)
            self.llbutton = tk.Button(
                navbuttons,
                text='<<< {0} PCs'.format(4 * self.nimages),
                command=lambda: self.changePCpage(self.PCpage - 4))
            self.lbutton = tk.Button(
                navbuttons,
                text='<<< {0} PCs'.format(self.nimages),
                command=lambda: self.changePCpage(self.PCpage - 1))
            self.homebutton = tk.Button(navbuttons,
                                        text='Home',
                                        command=lambda: self.changePCpage(0))
            self.rbutton = tk.Button(
                navbuttons,
                text='{0} PCs >>>'.format(self.nimages),
                command=lambda: self.changePCpage(self.PCpage + 1))
            self.rrbutton = tk.Button(
                navbuttons,
                text='{0} PCs >>>'.format(4 * self.nimages),
                command=lambda: self.changePCpage(self.PCpage + 4))

            # Make frame for pc max value controller
            pccontrol = tk.Frame(self)
            upper_limit = tk.StringVar()
            upper_limit.set(str(maxval))
            maxentry = tk.Entry(pccontrol, textvariable=upper_limit, width=5)
            entrylabel = tk.Label(pccontrol, text='Noise Floor:')

            # place the title
            label.grid(column=0, row=0)

            # place the navigation buttons and load panel
            self.llbutton.grid(column=0, row=0)
            self.lbutton.grid(column=1, row=0)
            self.homebutton.grid(column=2, row=0)
            self.rbutton.grid(column=3, row=0)
            self.rrbutton.grid(column=4, row=0)
            navbuttons.grid(column=0, row=2)

            # place max pc control panel
            entrylabel.grid(column=0, row=0)
            maxentry.grid(column=1, row=0)
            pccontrol.grid(column=0, row=3)

            self.loadPCpage()

        def callback_manager(self, event):
            if event.keysym == 'Right':
                if len(toggle) - self.nimages * (self.PCpage + 1) > 0:
                    self.changePCpage(self.PCpage + 1)
            elif event.keysym == 'Left':
                if self.PCpage != 0:
                    self.changePCpage(self.PCpage - 1)
            else:
                print('No callback defined for:', event.keysym)

        class PCframe(tk.Frame):
            # Create a frame to hold all eigenbrains
            def __init__(self, parent, indices):
                tk.Frame.__init__(self, parent)

                # variables to hold image buttons
                self.ncol = 5
                self.PCplotframe = tk.Frame(self, borderwidth=2)
                self.imagebutton = []

                # initialize and grid image buttons
                for i, n in enumerate(indices):
                    frame = (tk.Frame(self.PCplotframe))
                    self.imagebutton.append(self.imButton(frame, n))
                    c = i % self.ncol
                    r = i // self.ncol
                    frame.grid(row=r, column=c)

                self.PCplotframe.grid(column=0, row=1)  # grid main frame

            def update(self, parent, indices):
                # update each image button to contain indices given

                imagebutton = self.imagebutton

                #if there isn't a full set of indices, append None
                if len(indices) < len(imagebutton):
                    print('Not enough indices to fill page')
                    lendiff = len(imagebutton) - len(indices)
                    indices.extend([None] * lendiff)

                # update each image with new PC index
                for i, buttonhandle in enumerate(imagebutton):
                    buttonhandle.update(indices[i])

            class imButton(tk.Frame):
                # Image button.  When clicked, toggle[index] is switched.
                # Colormap is also switched to see status of PC
                def __init__(self, parent, pc_id):
                    self.pc_id = pc_id
                    f = Figure(figsize=(3, 3), dpi=100, frameon=False)
                    self.ax = f.add_subplot(111)
                    f.subplots_adjust(left=0.05,
                                      bottom=0.05,
                                      right=0.98,
                                      top=0.98)
                    self.canvas = FigureCanvasTkAgg(f, parent)
                    self.canvas.get_tk_widget().grid(row=1, column=1)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))

                    self.imgplot = self.ax.imshow(np.zeros((x, y)))
                    self.ax.axis('off')

                    self.update(pc_id)

                def update(self, pc_id):
                    self.pc_id = pc_id
                    self.ax.cla()
                    self.ax.axis('off')

                    if pc_id is None:  # clear image
                        im = np.empty((x, y))
                        im[:] = np.NAN
                        self.imgplot = self.ax.imshow(im)
                        self.canvas.draw()
                        return ()

                    eigenbrain = rebuild_eigenbrain(eig_vec, pc_id, roimask,
                                                    eigb_shape, maskind)
                    if rotate > 0:
                        eigenbrain = np.rot90(eigenbrain, rotate)
                    mean = np.nanmean(eigenbrain)
                    std = np.abs(np.nanstd(eigenbrain))

                    self.imgplot = self.ax.imshow(eigenbrain,
                                                  cmap=corr_cmap,
                                                  vmin=mean - 4 * std,
                                                  vmax=mean + 4 * std)

                    if toggle[pc_id] == 0:
                        self.imgplot.set_cmap(component_cmap)
                    else:
                        self.imgplot.set_cmap('Greys')

                    self.ax.annotate('Component {0}'.format(pc_id),
                                     color='grey',
                                     fontsize=10,
                                     xy=(1, 1))

                    self.canvas.draw()

                def on_key_press(self, event):
                    # callback for clicks on image.
                    # Change color and toggle value
                    try:
                        if toggle[self.pc_id] == 0:
                            self.imgplot.set_cmap('Greys_r')
                            toggle[self.pc_id] = 1
                        elif toggle[self.pc_id] == 1:
                            self.imgplot.set_cmap(component_cmap)
                            toggle[self.pc_id] = 0
                        self.canvas.draw()
                    except:
                        print('Index was out of range')

        def changePCpage(self, newpage):
            # callback for buttons to change page
            self.PCpage = newpage
            self.loadPCpage()

        def loadPCpage(self):
            # load a PC page based on self.PCpage

            # Reenable all buttons, disable any that shouldn't be allowed
            self.llbutton.config(state=['normal'])
            self.lbutton.config(state=['normal'])
            self.homebutton.config(state=['normal'])
            self.rbutton.config(state=['normal'])
            self.rrbutton.config(state=['normal'])

            if self.PCpage == 0:  # if on start page, disable home, -n
                self.homebutton.config(state=['disabled'])
                self.lbutton.config(state=['disabled'])
                self.llbutton.config(state=['disabled'])
            elif self.PCpage < 4:
                self.llbutton.config(state=['disabled'])

            if len(toggle) - (self.nimages * (self.PCpage + 1)) <= 0:
                # if total images - next page's max <=0, disable + n
                self.rbutton.config(state=['disabled'])
                self.rrbutton.config(state=['disabled'])
            elif len(toggle) - self.nimages * self.PCpage \
                    - 4*self.nimages <= 0:
                self.rrbutton.config(state=['disabled'])

            # Refresh the PC Grid
            startPC = self.PCpage * self.nimages
            endPC = startPC + self.nimages
            if endPC > len(toggle):
                endPC = len(toggle)
            PCindices = list(range(startPC, endPC))
            if hasattr(self, 'PCfigure'):
                self.PCfigure.update(self, PCindices)
                self.PCfigure.grid(column=0, row=1)
            else:
                self.PCfigure = self.PCframe(self, PCindices)
                self.PCfigure.grid(column=0, row=1)
                # grid PCfigure into StartPage

    class PCinfo(tk.Frame):
        # Page for viewing information about individual PCs

        def __init__(self, parent, controller):
            # Change the PC text box using the +/- keys.
            # This triggers updatePCvalue to run

            def updatePCval_button(delta):
                newval = int(self.selected_pc.get()) + delta
                self.selected_pc.set(newval)

            # Change the PC index using the text box
            def updatePCval():
                newvalue = self.selected_pc.get()
                print('Changing PC index to: {0}'.format(newvalue))

                if newvalue == '':  # empty value
                    print('Text box is blank.  Not updating')
                else:
                    try:  # make sure entry is an int, inside range of PCs
                        assert int(newvalue) < n_components - 1, (
                            'Index exceeds range')
                        assert int(newvalue) >= 0, 'Index below 0'
                        self.pc_id[0] = int(newvalue)
                        self.selected_pc.set(str(self.pc_id[0]))
                        fig.updateFigures(self.pc_id[0])

                    except:
                        print('Not changing upper PC cutoff.')
                        self.selected_pc.set(str(self.pc_id[0]))
                        # reset text box to previous value

            # Initialize PC info page
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text='Component Viewer:', font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            # Two components to page:
            pcviewer = tk.Frame(self)
            # grid of figures about selected PC
            pc_toolbar = tk.Frame(pcviewer)
            # toolbar for selecting PC index

            # Make PC selection toolbar
            self.pc_id = [0]
            self.selected_pc = tk.StringVar()
            self.selected_pc.set(str(self.pc_id[0]))
            pc_entry = tk.Entry(pc_toolbar,
                                textvariable=self.selected_pc,
                                width=5)
            self.selected_pc.trace('w',
                                   lambda nm, idx, mode, var=0: updatePCval())
            pc_entry_label = tk.Label(pc_toolbar, text='Component:')

            pm_toolbar = tk.Frame(pcviewer)

            inc = tk.Button(pm_toolbar,
                            text='+',
                            command=lambda: updatePCval_button(1))
            dec = tk.Button(pm_toolbar,
                            text='-',
                            command=lambda: updatePCval_button(-1))

            # grid pc selector frame
            pc_entry_label.grid(column=0, row=0)
            pc_entry.grid(column=1, row=0)
            inc.grid(column=0, row=0)
            dec.grid(column=1, row=0)

            # grid pcviewer frame
            pc_toolbar.pack()
            pm_toolbar.pack()
            pcviewer.pack()
            fig = self.PCfigures(self, self.pc_id)
            fig.pack()

        def callback_manager(self, event):
            if event.keysym == 'Right':
                newval = int(self.selected_pc.get()) + 1
                self.selected_pc.set(newval)
            elif event.keysym == 'Left':
                newval = int(self.selected_pc.get()) - 1
                self.selected_pc.set(newval)
            else:
                print('No callback defined for:', event.keysym)

        class PCfigures(tk.Frame):
            # Create class to hold and update all figures
            def __init__(self, parent, controller):

                # Create main frame, child of PCinfo page
                tk.Frame.__init__(self, parent)

                ncol = 2  # number of columns of figures
                self.figures = {}

                # List desired figures
                figlist = [
                    self.pcImage, self.timeCourse, self.fourierWaveletHistogram,
                    self.waveletSpectrum
                ]

                for i, Fig in enumerate(figlist):
                    # Not being used: self.fourierTimecourse
                    # self.fourierHistogram
                    figure = Fig(self)  # initialize each figure
                    c = i % ncol
                    r = i // ncol
                    figure.grid(row=r, column=c)  # grid it
                    self.figures[Fig] = figure
                    # store each handle in self.figures

                # initialize figures for PC #0
                self.updateFigures(0)

            def updateFigures(self, pc_id):
                # Update all figures in self.figures
                self.timecourse = timecourses[pc_id]

                eigenbrain = rebuild_eigenbrain(eig_vec, pc_id, roimask,
                                                eigb_shape, maskind)
                if rotate > 0:
                    eigenbrain = np.rot90(eigenbrain, rotate)
                mean = np.nanmean(eigenbrain)
                std = np.abs(np.nanstd(eigenbrain))

                eigenbrain[np.where(np.isnan(eigenbrain))] = 0
                self.eigenbrain = eigenbrain

                # Wavelet analysis
                wavelet = waveletAnalysis(self.timecourse.astype('float64'),
                                          fps=10,
                                          siglvl=0.95)

                # Dict to store all info for updating figures
                pc_variables = {
                    'pc_id': pc_id,
                    'timecourse': self.timecourse,
                    'wavelet': wavelet
                }

                # Update all figures
                # COMMENTHERE
                for i, Fig in enumerate(self.figures):
                    handle = self.figures[Fig]
                    handle.update(pc_variables)

            class pcImage(tk.Frame):
                # View eigenbrain (same as PCpage figures)
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.fig = f
                    self.ax = f.add_subplot(111)

                    self.ax.imshow(np.zeros((x, y)), cmap='Greys')
                    self.ax.axis('off')
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    pc_id = pc_variables['pc_id']
                    self.ax.cla()

                    eigenbrain = rebuild_eigenbrain(eig_vec, pc_id, roimask,
                                                    eigb_shape, maskind)
                    if rotate > 0:
                        eigenbrain = np.rot90(eigenbrain, rotate)
                    mean = np.nanmean(eigenbrain)
                    std = np.abs(np.nanstd(eigenbrain))

                    self.imgplot = self.ax.imshow(eigenbrain,
                                                  cmap=component_cmap,
                                                  vmin=mean - 4 * std,
                                                  vmax=mean + 4 * std)

                    self.ax.axis('off')
                    self.canvas.draw()

            class timeCourse(tk.Frame):
                # view timecourse of PC
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.fig = f
                    self.ax = f.add_subplot(111)
                    f.subplots_adjust(left=0.2,
                                      bottom=0.15,
                                      right=0.85,
                                      top=0.85)
                    self.ax.set_title('PC Timecourse')
                    self.ax.set_xlabel('Time (s)')
                    self.ax.set_ylabel('Sum Intensity (dFoF)')
                    self.ax.plot([])
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    timecourse = pc_variables['timecourse']
                    self.ax.lines.pop(0)
                    self.ax.plot(
                        np.arange(timecourse.size) / 10, timecourse, 'k')
                    self.canvas.draw()

            class timeCoursePCxCorr(tk.Frame):
                # view correlation between this PC and others
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.fig = f
                    self.ax = f.add_subplot(111)
                    f.subplots_adjust(left=0.2,
                                      bottom=0.15,
                                      right=0.85,
                                      top=0.85)
                    self.ax.set_title('Timecourse Correlations')
                    self.ax.set_xlabel('Component Number')
                    self.ax.set_ylabel("Pearson's Correlation")
                    self.ax.plot([])
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    pc_id = pc_variables['pc_id']
                    print('loading correlations for component #', pc_id)
                    correlation = timecourses[pc_id]

                    temp = np.copy(correlation)
                    temp[pc_id] = 0
                    temp[-1] = 0
                    ylim = np.std(temp)

                    self.ax.lines.pop(0)
                    self.ax.plot(correlation, 'k')
                    self.ax.set_ylim([-3 * ylim, 3 * ylim])
                    self.canvas.draw()

            class fourierTimecourse(tk.Frame):
                # Look at the Windowed Fourier Transform of the
                # PC timecourse
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.ax = f.add_subplot(111)
                    f.subplots_adjust(left=0.2,
                                      bottom=0.15,
                                      right=0.85,
                                      top=0.85)
                    self.ax.imshow(np.zeros((x, y)))
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    timecourse = pc_variables['timecourse']
                    timecourse = timecourse - timecourse.mean()
                    stft, fps, nyq, maxData = short_time_fourier_transform(
                        timecourse)

                    self.ax.cla()
                    self.ax.imshow(stft,
                                   cmap='jet',
                                   interpolation='nearest',
                                   aspect='auto',
                                   extent=(0, timecourse.size / fps, 0, nyq),
                                   clim=(0.0, maxData / 6))
                    self.ax.set_title('Sliding Fourier Transform')
                    self.ax.set_xlabel('Time (sec)')
                    self.ax.set_ylabel('Frequency (Hz)')
                    self.canvas.draw()

            class fourierHistogram(tk.Frame):
                # Look at the power histogram of the Windowed
                # Fourier Transform
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.fig = f
                    self.ax = f.add_subplot(111)
                    f.subplots_adjust(left=0.2,
                                      bottom=0.15,
                                      right=0.85,
                                      top=0.85)
                    self.ax.plot([])
                    # self.ax.set_xlim([0, 5])
                    self.ax.set_xlabel('Frequency (Hz)')
                    self.ax.set_ylabel('Normalized Power')
                    self.ax.set_title('Fourier Histogram')
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    timecourse = pc_variables['timecourse']
                    timecourse = timecourse - timecourse.mean()
                    stft, fps, nyq, maxData = short_time_fourier_transform(
                        timecourse)

                    stfthist = stft.sum(1)
                    self.ax.lines.pop(0)
                    self.ax.plot((np.arange(stfthist.size) / 10)[::-1],
                                 stfthist, 'k')
                    self.ax.set_yscale('log')
                    self.canvas.draw()

            class fourierWaveletHistogram(tk.Frame):
                # Look at the power histogram of the Windowed
                # Fourier Transform
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.fig = f
                    self.ax1 = f.add_subplot(111)
                    f.subplots_adjust(left=0.2,
                                      bottom=0.15,
                                      right=0.85,
                                      top=0.85)
                    self.ax1.set_xlim([0, 5])
                    self.ax1.plot([])
                    self.ax1.set_ylabel('Normalized Wavelet Power', color='r')
                    self.ax1.tick_params('y', colors='r')

                    self.ax2 = self.ax1.twinx()
                    self.ax2.plot([])
                    self.ax2.set_xlabel('Frequency (Hz)')
                    self.ax2.set_ylabel('Normalized Fourier Power', color='b')
                    self.ax2.tick_params('y', colors='b')
                    self.ax2.set_title('Fourier and Wavelet Histograms')

                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    linetype = ['-', '-.', '--', ':']
                    timecourse = pc_variables['timecourse']
                    timecourse = timecourse - timecourse.mean()
                    stft, fps, nyq, maxData = short_time_fourier_transform(
                        timecourse)
                    stfthist = stft.sum(1)

                    wavelet = pc_variables['wavelet']
                    wavelet.globalWaveletSpectrum()

                    for i in range(len(self.ax1.lines)):
                        self.ax1.lines.pop(0)
                    for i in range(len(self.ax2.lines)):
                        self.ax2.lines.pop(0)
                    # self.ax2.cla()
                    self.ax1.plot(wavelet.flambda, wavelet.gws, 'r')
                    self.ax1.plot(wavelet.flambda,
                                  wavelet.gws_sig,
                                  label='.95 sig wave.',
                                  ls=linetype[1],
                                  color='k')
                    self.ax1.set_yscale('log')
                    a, b = self.ax1.get_ylim()
                    self.ax1.legend()

                    freq = (np.arange(stfthist.size) / 10)[::-1]
                    self.ax2.plot(freq, stfthist, 'b')
                    self.ax2.set_yscale('log')
                    self.ax2.set_ylim([a, b])
                    self.ax2.set_xlim([0, 5])

                    self.canvas.draw()

            class waveletSpectrum(tk.Frame):
                # View wavelet power spectrum
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)

                    frame = tk.Frame(self)
                    f = Figure(figsize=(4, 4), dpi=100, frameon=False)
                    self.fig = f
                    self.ax = f.add_subplot(111)
                    f.subplots_adjust(left=0.2,
                                      bottom=0.15,
                                      right=0.85,
                                      top=0.85)
                    self.ax.plot([])
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):
                    if event.button == 3:
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    wavelet = pc_variables['wavelet']
                    self.ax.cla()
                    wavelet.plotPower(ax=self.ax)
                    self.canvas.draw()

    class DomainROIs(tk.Frame):
        # Page for viewing information about functional correlation of ICA domains

        def __init__(self, parent, controller):

            # Change the PC text box using the +/- keys.
            # This triggers updatePCvalue to run
            def updatePCval_button(delta):
                newval = int(self.selected_pc.get()) + delta
                self.selected_pc.set(newval)

            # Change the PC index using the text box
            def updatePCval():
                newvalue = self.selected_pc.get()
                print('\nChanging component index to: {0}'.format(newvalue))

                if newvalue == '':  # empty value
                    print('Text box is blank.  Not updating')
                else:
                    try:  # make sure entry is an int, inside range of PCs
                        assert int(newvalue) < n_components - 1, (
                            'Index exceeds range')
                        assert int(newvalue) >= 0, 'Index below 0'
                        self.pc_id[0] = int(newvalue)
                        self.selected_pc.set(str(self.pc_id[0]))
                        fig.updateFigures(self.pc_id[0])

                    except Exception as e:
                        print('Error!')
                        print('\t', e)
                        print('Not changing upper PC cutoff.')
                        self.selected_pc.set(str(self.pc_id[0]))
                        # reset text box to previous value

            # Initialize PC info page
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text='Domain ROI Viewer:', font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            # Two components to page:
            domain_viewer = tk.Frame(self)
            # grid of figures about selected PC
            pc_toolbar = tk.Frame(domain_viewer)
            # toolbar for selecting PC index

            # Make PC selection toolbar
            self.pc_id = [0]
            self.selected_pc = tk.StringVar()
            self.selected_pc.set(str(self.pc_id[0]))
            pc_entry = tk.Entry(pc_toolbar,
                                textvariable=self.selected_pc,
                                width=5)
            self.selected_pc.trace('w',
                                   lambda nm, idx, mode, var=0: updatePCval())
            pc_entry_label = tk.Label(pc_toolbar, text='Component ROI:')

            pm_toolbar = tk.Frame(domain_viewer)

            inc = tk.Button(pm_toolbar,
                            text='+',
                            command=lambda: updatePCval_button(1))
            dec = tk.Button(pm_toolbar,
                            text='-',
                            command=lambda: updatePCval_button(-1))

            # grid pc selector frame
            pc_entry_label.grid(column=0, row=0)
            pc_entry.grid(column=1, row=0)
            inc.grid(column=0, row=0)
            dec.grid(column=1, row=0)

            # grid domain_viewer frame
            pc_toolbar.pack()
            pm_toolbar.pack()
            # update_noise.pack()
            domain_viewer.pack()
            fig = self.DomainFigures(self, self.pc_id)
            fig.pack()

        def callback_manager(self, event):
            if event.keysym == 'Right':
                newval = int(self.selected_pc.get()) + 1
                self.selected_pc.set(newval)
            elif event.keysym == 'Left':
                newval = int(self.selected_pc.get()) - 1
                self.selected_pc.set(newval)
            else:
                print('No callback defined for:', event.keysym)

        class DomainFigures(tk.Frame):
            # Create class to hold and update all figures
            def __init__(self, parent, controller):

                # Create main frame, child of PCinfo page
                tk.Frame.__init__(self, parent)

                ncol = 2  # number of columns of figures
                self.figures = {}

                figlist = [self.domain_xcorr]

                for i, Fig in enumerate(figlist):
                    # Not being used: self.fourierTimecourse

                    figure = Fig(self)  # initialize each figure
                    c = i % ncol
                    r = i // ncol
                    figure.grid(row=r, column=c)  # grid it
                    self.figures[Fig] = figure
                    # store each handle in self.figures

                self.loaded = False

            def updateFigures(self, pc_id):
                # Update all figures in self.figures
                print('\nUpdating Domain Figures...')

                if not self.loaded:

                    if set(['domain_ROIs',
                            'ROI_timecourses']).issubset(components.keys()):
                        print('domain_ROIs found in components')
                        domain_ROIs = components['domain_ROIs']

                        if rotate > 0:
                            domain_ROIs = np.rot90(domain_ROIs, rotate)

                        xcorr = np.corrcoef(components['ROI_timecourses'] + \
                            components['mean_filtered'])

                        self.domain_ROIs = domain_ROIs
                        self.loaded = True
                        self.xcorr = xcorr
                        self.loaded = True

                    else:
                        print('No ROI timecourses found!')
                        self.loaded = False
                        self.domain_ROIs = None
                        self.domain_ROIs = None
                        self.xcorr = None
                        return  # dont try updating figures.

                else:
                    print('ROI information already loaded.')

                corr = self.xcorr[pc_id]

                # Dict to store all info for updating figures
                pc_variables = {
                    'corr': corr,
                    'pc_id': pc_id,
                    'domain_ROIs': self.domain_ROIs
                }

                # Update all figures
                for i, Fig in enumerate(self.figures):
                    handle = self.figures[Fig]
                    handle.update(pc_variables)

            class domain_xcorr(tk.Frame):
                # View eigenbrain (same as PCpage figures)
                def __init__(self, parent):
                    tk.Frame.__init__(self, parent)
                    self.parent = parent

                    frame = tk.Frame(self)
                    f = Figure(figsize=(8, 8), dpi=100, frameon=False)
                    f.subplots_adjust(top=1,
                                      bottom=0,
                                      right=1,
                                      left=0,
                                      hspace=0,
                                      wspace=0)
                    self.fig = f
                    self.ax = f.add_subplot(111)

                    img = np.zeros((x, y))
                    if roimask is not None:
                        rot_roimask = np.rot90(roimask, rotate)
                        img = np.rot90(img, rotate)
                        img[np.where(rot_roimask == 0)] = np.nan

                    cax = self.ax.imshow(img, vmin=-0.5, vmax=1, cmap=corr_cmap)
                    f.colorbar(cax)
                    self.ax.axis('off')
                    self.canvas = FigureCanvasTkAgg(f, frame)
                    self.canvas.mpl_connect(
                        'button_press_event',
                        lambda event: self.on_key_press(event))
                    self.canvas.draw()
                    self.canvas.get_tk_widget().pack()
                    frame.pack()

                def on_key_press(self, event):

                    if event.button == 1:  # if left click
                        if self.parent.loaded:

                            x = int(event.xdata)
                            y = int(event.ydata)

                            if roimask is not None:
                                rot_roimask = np.rot90(roimask, rotate)
                                if rot_roimask[y, x] != 1:
                                    # if click within brain
                                    return

                            pc_id = self.parent.domain_ROIs[y,
                                                            x].astype('uint16')

                            print('pc_id:', pc_id)
                            pc_id = pc_id
                            self.parent.updateFigures(pc_id)

                    elif event.button == 3:
                        print('right click: save figure.')
                        saveFigure(self.fig)

                def update(self, pc_variables):
                    print('updating xcorr...')
                    pc_id = pc_variables['pc_id']
                    domain_ROIs = pc_variables['domain_ROIs']
                    corr = pc_variables['corr']

                    if corr is not None:
                        print('getting map...')

                    else:
                        print('noise component: not displaying correlation')

                    self.ax.cla()
                    if self.parent.loaded:
                        frame = domain_map(domain_ROIs, values=corr)
                        self.ax.imshow(frame, vmin=-0.5, vmax=1, cmap=corr_cmap)
                    else:
                        frame = domain_map(domain_ROIs)
                        self.ax.imshow(frame, vmin=-0.5, vmax=1, cmap='gray')
                    self.ax.set_title('PC' + str(pc_id) +
                                      ': Domain Correlation')
                    self.ax.axis('off')
                    self.canvas.draw()

    class DomainRegions(tk.Frame):
        # Page for assigning regions of ICA domains

        def __init__(self, parent, controller):

            # Initialize PC info page
            tk.Frame.__init__(self, parent)
            label = tk.Label(self,
                             text='Domain Region Assignment:',
                             font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            self.selected_region = tk.StringVar()

            frame = tk.Frame(self)
            figure = tk.Frame(frame)
            fig = self.region_assignment_page(figure, self.selected_region)
            fig.pack()
            figure.grid(row=1, column=1)

            radioentry = tk.Frame(frame)
            radiolabel = tk.Label(radioentry, text='Region:', font=LARGE_FONT)
            radiolabel.pack(pady=10, padx=10)

            for name, value in regions:
                tk.Radiobutton(radioentry,
                               text=name,
                               variable=self.selected_region,
                               indicatoron=0,
                               value=value).pack(fill='x',
                                                 side='top',
                                                 anchor=tk.W)
            radioentry.grid(row=1, column=2)

            frame.pack()

        def callback_manager(self, event):

            regionkey = self.selected_region.get()
            if regionkey == '':
                regionkey = 0
            regionkey = int(regionkey)

            if (event.keysym == 'Right') | (event.keysym == 'Down'):
                if regionkey < len(regions):
                    self.selected_region.set(str(regionkey + 1))

            elif (event.keysym == 'Left') | (event.keysym == 'Up'):
                if regionkey > 1:
                    self.selected_region.set(str(regionkey - 1))

            else:
                print('No callback defined for:', event.keysym)

        class region_assignment_page(tk.Frame):
            # View eigenbrain (same as PCpage figures)
            def __init__(self, parent, entryvar):
                tk.Frame.__init__(self, parent)
                self.entryvar = entryvar

                frame = tk.Frame(self)
                f = Figure(figsize=(9, 9), dpi=100, frameon=False)
                f.subplots_adjust(top=1,
                                  bottom=0,
                                  right=1,
                                  left=0,
                                  hspace=0,
                                  wspace=0)
                self.fig = f
                self.ax = f.add_subplot(111)
                self.ax.imshow(np.zeros((x, y)))
                self.ax.axis('off')

                self.canvas = FigureCanvasTkAgg(f, frame)
                self.canvas.mpl_connect('button_press_event',
                                        lambda event: self.on_key_press(event))
                self.canvas.draw()
                self.canvas.get_tk_widget().pack()
                frame.pack()
                self.loaded = False

                if 'domain_ROIs' in components.keys():
                    print('n domains',
                          np.unique(domain_ROIs[~np.isnan(domain_ROIs)]).size)
                    self.edges = get_domain_edges(
                        domain_ROIs, linepad=5, clear_bg=True).astype('float64')
                    self.dmap = domain_map(domain_ROIs,
                                           values=region_assignment)
                    self.loaded = True
                    self.update()
                else:
                    self.loaded = False

            def on_key_press(self, event):

                if event.button == 1:  # if left click

                    region = self.entryvar.get()
                    if (region != '') and self.loaded:

                        x = int(event.xdata)
                        y = int(event.ydata)

                        if roimask is not None:
                            rot_roimask = np.rot90(roimask, rotate)
                            if rot_roimask[y, x] != 1:
                                # if click within brain
                                return

                        pc_id = domain_ROIs[y, x].astype('uint16')

                        print('clicked on id:', pc_id)
                        ind = np.where(domain_ROIs == pc_id)
                        region_assignment[pc_id] = float(region)
                        self.dmap[ind] = float(region)

                        self.update()

                elif event.button == 3:
                    print('right click: save figure.')
                    saveFigure(self.fig)

            def update(self):
                if self.loaded:
                    print('updating!')
                    self.ax.cla()
                    self.ax.imshow(self.dmap, cmap=region_cm, vmin=1, vmax=6)
                    self.ax.imshow(self.edges, vmin=0, vmax=2, cmap='binary')
                    self.ax.axis('off')
                    self.canvas.draw()
                else:
                    print('data not loaded.  not updating.')

    class PCautocorr(tk.Frame):
        # Page for viewing statistical information about the PC projection
        def __init__(self, parent, controller):

            # Initialize Stats Info page
            tk.Frame.__init__(self, parent)
            label = tk.Label(self,
                             text='Autocorrelation Viewer:',
                             font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            # Make frame for pc max value controller
            self.xlim_max = [maxval]  # maximum x limit for figures
            self.xlim_min = [0]
            pc_control = tk.Frame(self)
            pc_adjust = tk.Frame(pc_control)

            # initialize min and max controllers
            self.upper_limit = tk.StringVar()
            self.upper_limit.set(str(self.xlim_max[0]))
            self.upper_limit.trace(
                'w',
                lambda nm, idx, mode, var=self.upper_limit: self.updatemaxval())

            self.lower_limit = tk.StringVar()
            self.lower_limit.set(str(self.xlim_min[0]))
            self.lower_limit.trace(
                'w',
                lambda nm, idx, mode, var=self.lower_limit: self.updateminval())

            # horizontally pack label, pc_adjust into control vertically
            tk.Label(pc_control, text='PC axes:').pack()

            # pack elements in pc_adjust horizontally
            tk.Label(pc_adjust, text='from').pack(side=tk.LEFT)
            tk.Entry(pc_adjust, textvariable=self.lower_limit,
                     width=5).pack(side=tk.LEFT)
            tk.Label(pc_adjust, text='to').pack(side=tk.LEFT)
            tk.Entry(pc_adjust, textvariable=self.upper_limit,
                     width=5).pack(side=tk.LEFT)

            pc_adjust.pack()
            pc_control.pack()

            # statistics figures
            fig_grid = tk.Frame(self)
            self.statsviewer = self.AutoCorr(fig_grid)
            self.statsviewer.grid(row=0, column=0)

            fig_grid.pack()

        def updatemaxval(self):
            # Update the maximum principal component to be
            # included from textbox
            xlim_min = self.xlim_min
            xlim_max = self.xlim_max
            newvalue = self.upper_limit.get()
            print('Changing upper PC cutoff to: {0}'.format(newvalue))

            if newvalue == '':  # if blank, don't reset value
                self.upper_limit.set('')
            else:
                try:
                    assert int(
                        newvalue) <= n_components - 1, 'Index exceeds range'
                    assert int(newvalue) > xlim_min[0], 'Max index below min'
                    self.xlim_max[0] = int(newvalue)
                    self.updateFigures(xlim_min[0], xlim_max[0])
                except:  # not a valid number, reset to previous value
                    print('Not changing upper PC cutoff.')
                    # self.upper_limit.set(xlim_max[0])
                    print('upper limit before', self.upper_limit.get())
                    self.upper_limit.set(str(xlim_max[0]))
                    print('upper limit after', self.upper_limit.get())
                    # self.upper_limit.set('x')

        def updateminval(self):
            # Update the minimum principal component to be
            # included from textbox
            xlim_min = self.xlim_min
            xlim_max = self.xlim_max
            newvalue = self.lower_limit.get()
            print('Changing lower PC cutoff to: {0}'.format(newvalue))

            if newvalue == '':  # if blank, don't reset value
                self.lower_limit.set('x')
            else:
                try:
                    assert int(newvalue) < xlim_max[0], 'Index exceeds maximum'
                    assert int(newvalue) >= 0, 'Min index below 0'
                    self.xlim_min[0] = int(newvalue)
                    self.updateFigures(xlim_min[0], xlim_max[0])
                except:  # not a valid number, reset to previous value
                    print('Not changing lower PC cutoff.')
                    print('lower limit before', self.lower_limit.get())
                    self.lower_limit.set('x')
                    print('lower limit after', self.lower_limit.get())

        def updateFigures(self, xlim_min, xlim_max):
            # update all figures based on limits set by text update
            self.statsviewer.update(xlim_min, xlim_max)

            if type(timecourses) is np.ndarray:
                self.pcviewer.update(xlim_min, xlim_max)

        def callback_manager(self, event):
            if event.keysym == 'Right':
                newval = self.xlim_max[0] + 1
                self.upper_limit.set(newval)
            elif event.keysym == 'Left':
                newval = self.xlim_max[0] - 1
                self.upper_limit.set(newval)
            elif event.keysym == 'Up':
                newval = self.xlim_min[0] + 1
                self.lower_limit.set(newval)
            elif event.keysym == 'Down':
                newval = self.xlim_min[0] - 1
                self.lower_limit.set(newval)
            else:
                print('No callback defined for:', event.keysym)

        class AutoCorr(tk.Frame):
            # View variance explained by each eigenvalue
            def __init__(self, parent):
                tk.Frame.__init__(self, parent)

                # calculate lag autocorrelations
                x_grid = np.linspace(-0.2, 1.2, 1200)

                frame = tk.Frame(self)
                self.fig = Figure(figsize=(15, 8), dpi=100, frameon=False)
                f = self.fig

                self.ax = f.add_subplot(131)
                eig_std = components['timecourses'].std(axis=1)
                self.ax.plot(eig_std, '.r')
                print('cutoff:', cutoff)
                self.ax.plot(
                    np.where(lag1[:n_components] > cutoff)[0],
                    eig_std[np.where(lag1[:n_components] > cutoff)[0]], '.b')
                self.ax.set_xlabel('Independent Component')
                self.ax.set_ylabel('Timecourse Standard Deviation')

                self.ax = f.add_subplot(132)
                self.ax.set_ylim(-0.2, 1.2)
                f.subplots_adjust(left=0.2, bottom=0.15, right=0.85, top=0.85)
                self.ax.plot(np.where(lag1 > cutoff)[0],
                             lag1[lag1 > cutoff],
                             'b',
                             label='Lag-1 Signal')
                self.ax.plot(np.where(lag1 < cutoff)[0],
                             lag1[lag1 < cutoff],
                             'r',
                             label='Lag-1 Noise')
                self.ax.set_title('PC AutoCorrelation')
                self.ax.set_ylabel('AutoCorrelation')
                self.ax.set_xlabel('Independent Component')
                self.ax.legend()

                self.ax = f.add_subplot(133)
                self.ax.set_ylim(-0.2, 1.2)
                n, _, _ = self.ax.hist(lag1, bins=50, orientation='horizontal')
                self.ax.plot(np.exp(log_pdf) * n.max() / np.exp(log_pdf).max(),
                             x_grid,
                             lw=3)
                if cutoff is not None:
                    self.ax.axhline(cutoff, color='r', lw=3, linestyle='dashed')
                self.ax.set_title('Lag-1 Histogram')
                self.ax.set_xlabel('n')

                self.canvas = FigureCanvasTkAgg(f, frame)
                self.canvas.mpl_connect('button_press_event',
                                        lambda event: self.on_key_press(event))
                self.canvas.draw()
                self.canvas.get_tk_widget().pack()
                frame.pack()

            def update(self, xlim_min, xlim_max):
                self.ax.set_xlim([xlim_min, xlim_max])
                self.canvas.draw()

            def on_key_press(self, event):
                if event.button == 3:
                    saveFigure(self.fig)

    # Run main loop
    app = PCAgui()
    app.mainloop()
    app.quit()

    # Find which indices to use for reconstruction
    if toggle[0] == 100:
        raise Exception('Operation was cancelled')

    if toggle.sum() == 0:
        print('All components are classified as signal')
    else:
        print('{0} components classified as signal, {1} '
              'will be used for signal reconstruction.'.format(
                  toggle.sum(), int(toggle.size - toggle.sum())))

    #update components with toggle info
    components['artifact_components'] = toggle
    components['cutoff'] = cutoff
    components['noise_components'] = noise_components
    if 'domain_ROIs' in components:
        components['region_assignment'] = region_assignment
        components['region_labels'] = regions

    if load_hdf5:  # if data came from save file, append toggle and timecourses
        f.save({'artifact_components': toggle})
        f.save({'noise_components': noise_components})
        f.save({'timecourses': components['timecourses']})
        if 'domain_ROIs' in components:
            f.save({
                'region_assignment': region_assignment,
                'region_labels': regions
            })

    print('\n')
    return components
