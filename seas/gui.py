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


def run_gui(components: dict,
            rotate: int = 0,
            savepath: str = None,
            default_assignment: np.ndarray = None) -> dict:
    '''
    Create a tkinter GUI to select noise components from ica-decomposition components file.
    If hdf5 information is given using a string input to components, or a savepath which is a valid hdf5 file,
    component artifact assignment will be saved to the file.

    Returns toggle, a boolean array 
    of either True or False.  Components that have been determined to be 
    noise are 'True', components to keep are 'False'.
    Optional toggle input is a boolean of starting values 
    for noise_components.

    Arguments:
        components: A dictionary containing the experiment and decomposition information.
        rotate: An integer number of clockwise rotations.
        savepath: Where to save or load component information from.
        default_assignment: The default assignment of which components are signal or artifact.

    Returns:
        components: The components dict, updated with any new region assignment, or artifact assignment, if applicable. 
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

    # Load all components from dict.
    eig_vec = components['eig_vec']
    if 'thresholds' in components.keys():
        thresholds = components['thresholds']
    roimask = components['roimask']
    shape = components['shape']
    t, x, y = shape
    eigb_shape = (x, y)

    # Find number of components.
    n_components = eig_vec.shape[1]
    print('number of components:', n_components)

    # Start timecourses variable for storing rebuilt timecourses of components.
    if 'timecourses' in components:
        print('timecourses found')
        timecourses = components['timecourses']
    else:
        print('timecourses not found in components')
        print('Initializing empty timecourse vector')
        timecourses = components['timecourses']

    # Start noise_components variable for listing which components are noise.
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

    # Start toggle variable for checking which components shouldn't
    # be included.
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

    # Load mask indexing for faster rebuilding of eigenbrains.
    if roimask is not None:
        maskind = np.where(roimask.flat == 1)
    else:
        maskind = None

    region_cm = config['colormap']['domains']
    component_cmap = config['colormap']['components']
    corr_cmap = config['colormap']['correlation']

    regions = config['regions']

    # Convert from defaults dict to sorted list of tuples.
    keylist = []
    valuelist = []

    for key in regions:
        keylist.append(key)
        valuelist.append(regions[key])

    sortindex = [i[0] for i in sorted(enumerate(valuelist), key=lambda x: x[1])]

    regions = []
    for i in sortindex:
        regions.append((keylist[i], valuelist[i]))

    LARGE_FONT = ('Verdana', 12)

    toggle_save = [True]  # Default on for easy click-to-save figures.
    toggle_debug = [False]  # Default off -- don't load ICA pixel properties.

    def save_figure(fig_handle):
        print('figure was pressed!')
        # Callback for clicks on image.
        # Change color and toggle value.

        if toggle_save[0]:
            print('trying to save..')
            file_path = tk.filedialog.asksaveasfilename()
            print(file_path)

            if type(file_path) is str:  # If path was provided.
                # If there was no extension, add .png.
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

    # Create main application as tk.Tk.
    class AnalysisGui(tk.Tk):  # Note: Inherits from Tk class.

        def __init__(self, *args, **kwargs):
            tk.Tk.__init__(self, *args,
                           **kwargs)  # Initialize the parent tk class.
            tk.Tk.wm_title(self, 'Component Viewer')
            # Every page initializes a container to hold its content.
            container = tk.Frame(self)
            container.pack(side='top', fill='both', expand=True)

            # Set page expansion properties.
            container.grid_rowconfigure(0, weight=1)
            # Row/columns expand equally.
            container.grid_columnconfigure(0, weight=1)

            # Make the menu bar (top banner).
            menubar = tk.Menu(container)

            filemenu = tk.Menu(menubar, tearoff=0)
            filemenu.add_separator()
            filemenu.add_command(label='save', command=self.quit)
            filemenu.add_command(label='exit',
                                 command=lambda: self.cancelcallback(toggle))
            menubar.add_cascade(label='file', menu=filemenu)

            def toggle_figure_saving():
                toggle_save[0] = not toggle_save[0]

            def toggle_figure_saving():
                toggle_debug[0] = not toggle_debug[0]

            editmenu = tk.Menu(menubar, tearoff=0)
            editmenu.add_separator()
            editmenu.add_command(label='toggle figure saving',
                                 command=lambda: toggle_figure_saving())
            editmenu.add_command(label='toggle ica pixel debug',
                                 command=lambda: toggle_ica_debug())
            menubar.add_cascade(label='edit', menu=editmenu)

            pagemenu = tk.Menu(menubar, tearoff=1)
            pagemenu.add_separator()
            pagemenu.add_command(label='view components',
                                 command=lambda: self.show_frame(ComponentPage))
            pagemenu.add_command(label='Component information',
                                 command=lambda: self.show_frame(ComponentInfo))
            pagemenu.add_command(label='Domain Correlations',
                                 command=lambda: self.show_frame(DomainROIs))
            pagemenu.add_command(label='Domain Region Assignment',
                                 command=lambda: self.show_frame(DomainRegions))
            pagemenu.add_command(
                label='Domain Autocorrelations',
                command=lambda: self.show_frame(DomainAutoCorr))
            menubar.add_cascade(label='view', menu=pagemenu)

            tk.Tk.config(self, menu=menubar)

            # Create container to hold for all pages for page switching.
            self.frames = {}

            # List all pages here:
            for F in (ComponentPage, ComponentInfo, DomainROIs, DomainRegions,
                      DomainAutoCorr):
                frame = F(container, self)
                self.frames[F] = frame
                # set currently active frame to StartPage
                frame.grid(row=0, column=0, sticky='nsew')

            # Initialize default page.
            default_page = ComponentPage
            self.show_frame(default_page)

            # Global event binding commands.
            self.bind("<Escape>", lambda event: self.cancelcallback(toggle))
            self.bind("s", lambda event: self.quit())
            self.bind("<F1>", lambda event: self.show_frame(ComponentPage))
            self.bind("<F2>", lambda event: self.show_frame(ComponentInfo))
            self.bind("<F3>", lambda event: self.show_frame(DomainROIs))
            self.bind("<F4>", lambda event: self.show_frame(DomainRegions))
            self.bind("<F5>", lambda event: self.show_frame(DomainAutoCorr))

            # Use global focus to capture bindings,
            # send them to local callback manager.
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
            # Selects frame, raises it.
            frame = self.frames[cont]
            self.current_page = frame
            frame.tkraise()

        def cancelcallback(self, toggle):
            # Quits the app, doesn't rebuild the movie.
            print('Operation was cancelled.')
            toggle[0] = 100
            self.quit()

    # Create each frame and components as individual classes:
    class ComponentPage(tk.Frame):
        # Page to view all principal component images, toggle via clicks.

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)  # Parent is controller.

            # Create frame for each component, initialize component indices.
            self.n_images = 15
            self.current_page_number = 0

            # Initialize the title.
            label = tk.Label(self,
                             text='Select Components to Remove:',
                             font=LARGE_FONT)

            # Initialize navigation buttons.
            navbuttons = tk.Frame(self)
            self.llbutton = tk.Button(navbuttons,
                                      text='<<< {0} PCs'.format(4 *
                                                                self.n_images),
                                      command=lambda: self.changeComponentPage(
                                          self.current_page_number - 4))
            self.lbutton = tk.Button(navbuttons,
                                     text='<<< {0} PCs'.format(self.n_images),
                                     command=lambda: self.changeComponentPage(
                                         self.current_page_number - 1))
            self.homebutton = tk.Button(
                navbuttons,
                text='Home',
                command=lambda: self.changeComponentPage(0))
            self.rbutton = tk.Button(navbuttons,
                                     text='{0} PCs >>>'.format(self.n_images),
                                     command=lambda: self.changeComponentPage(
                                         self.current_page_number + 1))
            self.rrbutton = tk.Button(navbuttons,
                                      text='{0} PCs >>>'.format(4 *
                                                                self.n_images),
                                      command=lambda: self.changeComponentPage(
                                          self.current_page_number + 4))

            # Make frame for pc max value controller.
            pccontrol = tk.Frame(self)
            upper_limit = tk.StringVar()
            upper_limit.set(str(maxval))
            maxentry = tk.Entry(pccontrol, textvariable=upper_limit, width=5)
            entrylabel = tk.Label(pccontrol, text='Noise Floor:')

            # Place the title.
            label.grid(column=0, row=0)

            # Place the navigation buttons and load panel.
            self.llbutton.grid(column=0, row=0)
            self.lbutton.grid(column=1, row=0)
            self.homebutton.grid(column=2, row=0)
            self.rbutton.grid(column=3, row=0)
            self.rrbutton.grid(column=4, row=0)
            navbuttons.grid(column=0, row=2)

            # Place max pc control panel.
            entrylabel.grid(column=0, row=0)
            maxentry.grid(column=1, row=0)
            pccontrol.grid(column=0, row=3)

            self.loadComponentPage()

        def callback_manager(self, event):
            if event.keysym == 'Right':
                if len(toggle) - self.n_images * (self.current_page_number +
                                                  1) > 0:
                    self.changeComponentPage(self.current_page_number + 1)
            elif event.keysym == 'Left':
                if self.current_page_number != 0:
                    self.changeComponentPage(self.current_page_number - 1)
            else:
                print('No callback defined for:', event.keysym)

        class ComponentFrame(tk.Frame):
            # Create a frame to hold each component.
            def __init__(self, parent, indices):
                tk.Frame.__init__(self, parent)

                # Variables to hold image buttons.
                self.ncol = 5
                self.PCplotframe = tk.Frame(self, borderwidth=2)
                self.imagebutton = []

                # Initialize and grid image buttons.
                for i, n in enumerate(indices):
                    frame = (tk.Frame(self.PCplotframe))
                    self.imagebutton.append(self.imButton(frame, n))
                    c = i % self.ncol
                    r = i // self.ncol
                    frame.grid(row=r, column=c)

                self.PCplotframe.grid(column=0, row=1)  # grid main frame

            def update(self, parent, indices):
                # Update each image button to contain indices given.

                imagebutton = self.imagebutton

                # If there isn't a full set of indices, append None.
                if len(indices) < len(imagebutton):
                    print('Not enough indices to fill page')
                    lendiff = len(imagebutton) - len(indices)
                    indices.extend([None] * lendiff)

                # Update each image with new PC index.
                for i, buttonhandle in enumerate(imagebutton):
                    buttonhandle.update(indices[i])

            class imButton(tk.Frame):
                # Image button.  When clicked, toggle[index] is switched.
                # Colormap is also switched to see status of PC.
                def __init__(self, parent, component_id):
                    self.component_id = component_id
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

                    self.update(component_id)

                def update(self, component_id):
                    self.component_id = component_id
                    self.ax.cla()
                    self.ax.axis('off')

                    if component_id is None:  # Clear image.
                        im = np.empty((x, y))
                        im[:] = np.NAN
                        self.imgplot = self.ax.imshow(im)
                        self.canvas.draw()
                        return ()

                    eigenbrain = rebuild_eigenbrain(eig_vec, component_id,
                                                    roimask, eigb_shape,
                                                    maskind)
                    if rotate > 0:
                        eigenbrain = np.rot90(eigenbrain, rotate)
                    mean = np.nanmean(eigenbrain)
                    std = np.abs(np.nanstd(eigenbrain))

                    self.imgplot = self.ax.imshow(eigenbrain,
                                                  cmap=corr_cmap,
                                                  vmin=mean - 4 * std,
                                                  vmax=mean + 4 * std)

                    if toggle[component_id] == 0:
                        self.imgplot.set_cmap(component_cmap)
                    else:
                        self.imgplot.set_cmap('Greys')

                    self.ax.annotate('Component {0}'.format(component_id),
                                     color='grey',
                                     fontsize=10,
                                     xy=(1, 1))

                    self.canvas.draw()

                def on_key_press(self, event):
                    # callback for clicks on image.
                    # Change color and toggle value.
                    try:
                        if toggle[self.component_id] == 0:
                            self.imgplot.set_cmap('Greys_r')
                            toggle[self.component_id] = 1
                        elif toggle[self.component_id] == 1:
                            self.imgplot.set_cmap(component_cmap)
                            toggle[self.component_id] = 0
                        self.canvas.draw()
                    except:
                        print('Index was out of range')

        def changeComponentPage(self, newpage):
            # Callback for buttons to change page.
            self.current_page_number = newpage
            self.loadComponentPage()

        def loadComponentPage(self):
            # Load a PC page based on self.current_page_number.

            # Reenable all buttons, disable any that shouldn't be allowed.
            self.llbutton.config(state=['normal'])
            self.lbutton.config(state=['normal'])
            self.homebutton.config(state=['normal'])
            self.rbutton.config(state=['normal'])
            self.rrbutton.config(state=['normal'])

            if self.current_page_number == 0:
                # If on start page, disable home, -n.
                self.homebutton.config(state=['disabled'])
                self.lbutton.config(state=['disabled'])
                self.llbutton.config(state=['disabled'])
            elif self.current_page_number < 4:
                self.llbutton.config(state=['disabled'])

            if len(toggle) - (self.n_images *
                              (self.current_page_number + 1)) <= 0:
                # If total images - next page's max <=0, disable + n.
                self.rbutton.config(state=['disabled'])
                self.rrbutton.config(state=['disabled'])
            elif len(toggle) - self.n_images * self.current_page_number \
                    - 4*self.n_images <= 0:
                self.rrbutton.config(state=['disabled'])

            # Refresh the PC Grid.
            startPC = self.current_page_number * self.n_images
            endPC = startPC + self.n_images
            if endPC > len(toggle):
                endPC = len(toggle)
            PCindices = list(range(startPC, endPC))
            if hasattr(self, 'PCfigure'):
                self.PCfigure.update(self, PCindices)
                self.PCfigure.grid(column=0, row=1)
            else:
                self.PCfigure = self.ComponentFrame(self, PCindices)
                self.PCfigure.grid(column=0, row=1)
                # Grid PCfigure into StartPage.

    class ComponentInfo(tk.Frame):
        # Page for viewing information about individual Components.

        def __init__(self, parent, controller):
            # Change the component text box using the +/- keys.
            # This triggers update_component_index to run.

            def update_component_index_button(delta):
                newval = int(self.selected_component.get()) + delta
                self.selected_component.set(newval)

            # Change the component index using the text box.
            def update_component_index():
                newvalue = self.selected_component.get()
                print('Changing component index to: {0}'.format(newvalue))

                if newvalue == '':  # Empty value.
                    print('Text box is blank.  Not updating')
                else:
                    try:  # Make sure entry is an int, inside range of PCs.
                        assert int(newvalue) < n_components - 1, (
                            'Index exceeds range')
                        assert int(newvalue) >= 0, 'Index below 0'
                        self.component_id[0] = int(newvalue)
                        self.selected_component.set(str(self.component_id[0]))
                        fig.update_figures(self.component_id[0])

                    except:
                        print('Not changing upper PC cutoff.')
                        self.selected_component.set(str(self.component_id[0]))
                        # Reset text box to previous value.

            # Initialize PC info page.
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text='Component Viewer:', font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            # Two components to page:
            component_viewer_frame = tk.Frame(self)
            # Grid of figures about selected PC.
            current_component_toolbar = tk.Frame(component_viewer_frame)
            # Toolbar for selecting PC index.

            # Make PC selection toolbar.
            self.component_id = [0]
            self.selected_component = tk.StringVar()
            self.selected_component.set(str(self.component_id[0]))
            pc_entry = tk.Entry(current_component_toolbar,
                                textvariable=self.selected_component,
                                width=5)
            self.selected_component.trace(
                'w', lambda nm, idx, mode, var=0: update_component_index())
            pc_entry_label = tk.Label(current_component_toolbar,
                                      text='Component:')

            component_adjust_toolbar = tk.Frame(component_viewer_frame)

            inc = tk.Button(component_adjust_toolbar,
                            text='+',
                            command=lambda: update_component_index_button(1))
            dec = tk.Button(component_adjust_toolbar,
                            text='-',
                            command=lambda: update_component_index_button(-1))

            # Grid pc selector frame.
            pc_entry_label.grid(column=0, row=0)
            pc_entry.grid(column=1, row=0)
            inc.grid(column=0, row=0)
            dec.grid(column=1, row=0)

            # Grid component_viewer_frame frame.
            current_component_toolbar.pack()
            component_adjust_toolbar.pack()
            component_viewer_frame.pack()
            fig = self.ComponentFigures(self, self.component_id)
            fig.pack()

        def callback_manager(self, event):
            if event.keysym == 'Right':
                new_component_value = int(self.selected_component.get()) + 1
                self.selected_component.set(new_component_value)
            elif event.keysym == 'Left':
                new_component_value = int(self.selected_component.get()) - 1
                self.selected_component.set(new_component_value)
            else:
                print('No callback defined for:', event.keysym)

        class ComponentFigures(tk.Frame):
            # Create class to hold and update all figures.
            def __init__(self, parent, controller):

                # Create main frame, child of ComponentInfo page.
                tk.Frame.__init__(self, parent)

                ncol = 2  # Number of columns of figures.
                self.figures = {}

                # List desired figures.
                figlist = [
                    self.pcImage, self.timeCourse, self.FourierWaveletHistogram,
                    self.WaveletSpectrum
                ]

                for i, Fig in enumerate(figlist):
                    # Not being used: self.fourierTimecourse,
                    # self.fourierHistogram.
                    figure = Fig(self)  # Initialize each figure.
                    c = i % ncol
                    r = i // ncol
                    figure.grid(row=r, column=c)  # Grid it.
                    self.figures[Fig] = figure
                    # Store each handle in self.figures.

                # Initialize figures for PC #0.
                self.update_figures(0)

            def update_figures(self, component_id):
                # Update all figures in self.figures.
                self.timecourse = timecourses[component_id]

                eigenbrain = rebuild_eigenbrain(eig_vec, component_id, roimask,
                                                eigb_shape, maskind)
                if rotate > 0:
                    eigenbrain = np.rot90(eigenbrain, rotate)
                mean = np.nanmean(eigenbrain)
                std = np.abs(np.nanstd(eigenbrain))

                eigenbrain[np.where(np.isnan(eigenbrain))] = 0
                self.eigenbrain = eigenbrain

                # Wavelet analysis:
                wavelet = waveletAnalysis(self.timecourse.astype('float64'),
                                          fps=10,
                                          siglvl=0.95)

                # Dict to store all info for updating figures.
                component_variables = {
                    'component_id': component_id,
                    'timecourse': self.timecourse,
                    'wavelet': wavelet
                }

                # Update all figures.
                for i, Fig in enumerate(self.figures):
                    handle = self.figures[Fig]
                    handle.update(component_variables)

            class pcImage(tk.Frame):
                # View eigenbrain (same as ComponentPage figures).
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    component_id = component_variables['component_id']
                    self.ax.cla()

                    eigenbrain = rebuild_eigenbrain(eig_vec, component_id,
                                                    roimask, eigb_shape,
                                                    maskind)
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
                # View component timecourse.
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    timecourse = component_variables['timecourse']
                    self.ax.lines.pop(0)
                    self.ax.plot(
                        np.arange(timecourse.size) / 10, timecourse, 'k')
                    self.canvas.draw()

            class timeCoursePCxCorr(tk.Frame):
                # View correlation between this PC and others.
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    component_id = component_variables['component_id']
                    print('loading correlations for component #', component_id)
                    correlation = timecourses[component_id]

                    temp = np.copy(correlation)
                    temp[component_id] = 0
                    temp[-1] = 0
                    ylim = np.std(temp)

                    self.ax.lines.pop(0)
                    self.ax.plot(correlation, 'k')
                    self.ax.set_ylim([-3 * ylim, 3 * ylim])
                    self.canvas.draw()

            class fourierTimecourse(tk.Frame):
                # Look at the windowed fourier transform of the
                # component timecourse.
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    timecourse = component_variables['timecourse']
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
                # Look at the power histogram of the windowed
                # fourier transform.
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    timecourse = component_variables['timecourse']
                    timecourse = timecourse - timecourse.mean()
                    stft, fps, nyq, maxData = short_time_fourier_transform(
                        timecourse)

                    stfthist = stft.sum(1)
                    self.ax.lines.pop(0)
                    self.ax.plot((np.arange(stfthist.size) / 10)[::-1],
                                 stfthist, 'k')
                    self.ax.set_yscale('log')
                    self.canvas.draw()

            class FourierWaveletHistogram(tk.Frame):
                # Look at the power histogram of the windowed
                # fourier transform.
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    linetype = ['-', '-.', '--', ':']
                    timecourse = component_variables['timecourse']
                    timecourse = timecourse - timecourse.mean()
                    stft, fps, nyq, maxData = short_time_fourier_transform(
                        timecourse)
                    stfthist = stft.sum(1)

                    wavelet = component_variables['wavelet']
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

            class WaveletSpectrum(tk.Frame):
                # View wavelet power spectrum.
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
                        save_figure(self.fig)

                def update(self, component_variables):
                    wavelet = component_variables['wavelet']
                    self.ax.cla()
                    wavelet.plotPower(ax=self.ax)
                    self.canvas.draw()

    class DomainROIs(tk.Frame):
        # Page for viewing information about functional correlation of ICA domains.

        def __init__(self, parent, controller):

            # Change the PC text box using the +/- keys.
            # This triggers update_component_index to run.
            def update_component_index_button(delta):
                newval = int(self.selected_component.get()) + delta
                self.selected_component.set(newval)

            # Change the PC index using the text box.
            def update_component_index():
                newvalue = self.selected_component.get()
                print('\nChanging component index to: {0}'.format(newvalue))

                if newvalue == '':  # Empty value.
                    print('Text box is blank.  Not updating')
                else:
                    try:  # Make sure entry is an int, inside range of PCs.
                        assert int(newvalue) < n_components - 1, (
                            'Index exceeds range')
                        assert int(newvalue) >= 0, 'Index below 0'
                        self.component_id[0] = int(newvalue)
                        self.selected_component.set(str(self.component_id[0]))
                        fig.update_figures(self.component_id[0])

                    except Exception as e:
                        print('Error!')
                        print('\t', e)
                        print('Not changing upper PC cutoff.')
                        self.selected_component.set(str(self.component_id[0]))
                        # Reset text box to previous value.

            # Initialize PC info page.
            tk.Frame.__init__(self, parent)
            label = tk.Label(self, text='Domain ROI Viewer:', font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            # Two components to page:
            domain_viewer = tk.Frame(self)
            # Grid of figures about selected PC.
            current_component_toolbar = tk.Frame(domain_viewer)
            # toolbar for selecting PC index

            # Make PC selection toolbar.
            self.component_id = [0]
            self.selected_component = tk.StringVar()
            self.selected_component.set(str(self.component_id[0]))
            pc_entry = tk.Entry(current_component_toolbar,
                                textvariable=self.selected_component,
                                width=5)
            self.selected_component.trace(
                'w', lambda nm, idx, mode, var=0: update_component_index())
            pc_entry_label = tk.Label(current_component_toolbar,
                                      text='Component ROI:')

            component_adjust_toolbar = tk.Frame(domain_viewer)

            inc = tk.Button(component_adjust_toolbar,
                            text='+',
                            command=lambda: update_component_index_button(1))
            dec = tk.Button(component_adjust_toolbar,
                            text='-',
                            command=lambda: update_component_index_button(-1))

            # Grid pc selector frame.
            pc_entry_label.grid(column=0, row=0)
            pc_entry.grid(column=1, row=0)
            inc.grid(column=0, row=0)
            dec.grid(column=1, row=0)

            # Grid domain_viewer frame.
            current_component_toolbar.pack()
            component_adjust_toolbar.pack()
            domain_viewer.pack()
            fig = self.DomainFigures(self, self.component_id)
            fig.pack()

        def callback_manager(self, event):
            if event.keysym == 'Right':
                newval = int(self.selected_component.get()) + 1
                self.selected_component.set(newval)
            elif event.keysym == 'Left':
                newval = int(self.selected_component.get()) - 1
                self.selected_component.set(newval)
            else:
                print('No callback defined for:', event.keysym)

        class DomainFigures(tk.Frame):
            # Create class to hold and update all figures.
            def __init__(self, parent, controller):

                # Create main frame, child of ComponentInfo page.
                tk.Frame.__init__(self, parent)

                ncol = 2  # Number of columns of figures.
                self.figures = {}

                figlist = [self.domain_xcorr]

                for i, Fig in enumerate(figlist):
                    # Not being used: self.fourierTimecourse.

                    figure = Fig(self)  # Initialize each figure.
                    c = i % ncol
                    r = i // ncol
                    figure.grid(row=r, column=c)  # Grid it.
                    self.figures[Fig] = figure
                    # Store each handle in self.figures.

                self.loaded = False

            def update_figures(self, component_id):
                # Update all figures in self.figures.
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
                        return  # Dont try updating figures.

                else:
                    print('ROI information already loaded.')

                corr = self.xcorr[component_id]

                # Dict to store all info for updating figures.
                component_variables = {
                    'corr': corr,
                    'component_id': component_id,
                    'domain_ROIs': self.domain_ROIs
                }

                # Update all figures.
                for i, Fig in enumerate(self.figures):
                    handle = self.figures[Fig]
                    handle.update(component_variables)

            class domain_xcorr(tk.Frame):
                # View eigenbrain (same as ComponentPage figures).
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

                    if event.button == 1:  # If left click.
                        if self.parent.loaded:

                            x = int(event.xdata)
                            y = int(event.ydata)

                            if roimask is not None:
                                rot_roimask = np.rot90(roimask, rotate)
                                if rot_roimask[y, x] != 1:
                                    # If click within brain.
                                    return

                            component_id = self.parent.domain_ROIs[y, x].astype(
                                'uint16')

                            print('component_id:', component_id)
                            component_id = component_id
                            self.parent.update_figures(component_id)

                    elif event.button == 3:
                        print('right click: save figure.')
                        save_figure(self.fig)

                def update(self, component_variables):
                    print('updating xcorr...')
                    component_id = component_variables['component_id']
                    domain_ROIs = component_variables['domain_ROIs']
                    corr = component_variables['corr']

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
                    self.ax.set_title('PC' + str(component_id) +
                                      ': Domain Correlation')
                    self.ax.axis('off')
                    self.canvas.draw()

    class DomainRegions(tk.Frame):
        # Page for assigning regions of ICA domains.

        def __init__(self, parent, controller):

            # Initialize page.
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
            # View component (same as ComponentPage figures).
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

                if event.button == 1:  # If left click.

                    region = self.entryvar.get()
                    if (region != '') and self.loaded:

                        x = int(event.xdata)
                        y = int(event.ydata)

                        if roimask is not None:
                            rot_roimask = np.rot90(roimask, rotate)
                            if rot_roimask[y, x] != 1:
                                # If click within brain.
                                return

                        component_id = domain_ROIs[y, x].astype('uint16')

                        print('clicked on id:', component_id)
                        ind = np.where(domain_ROIs == component_id)
                        region_assignment[component_id] = float(region)
                        self.dmap[ind] = float(region)

                        self.update()

                elif event.button == 3:
                    print('right click: save figure.')
                    save_figure(self.fig)

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

    class DomainAutoCorr(tk.Frame):
        # Page for viewing statistical information about the PC projection.
        def __init__(self, parent, controller):

            # Initialize Stats Info page.
            tk.Frame.__init__(self, parent)
            label = tk.Label(self,
                             text='Autocorrelation Viewer:',
                             font=LARGE_FONT)
            label.pack(pady=10, padx=10)

            # Make frame for pc max value controller.
            self.xlim_max = [maxval]  # Maximum x limit for figures.
            self.xlim_min = [0]
            pc_control = tk.Frame(self)
            pc_adjust = tk.Frame(pc_control)

            # Initialize min and max controllers.
            self.upper_limit = tk.StringVar()
            self.upper_limit.set(str(self.xlim_max[0]))
            self.upper_limit.trace(
                'w',
                lambda nm, idx, mode, var=self.upper_limit: self.updatemaxval())

            self.lower_limit = tk.StringVar()
            self.lower_limit.set(str(self.xlim_min[0]))
            self.lower_limit.trace('w',
                                   lambda nm, idx, mode, var=self.lower_limit:
                                   self.update_min_val())

            # Horizontally pack label, pc_adjust into control vertically.
            tk.Label(pc_control, text='PC axes:').pack()

            # Pack elements in pc_adjust horizontally.
            tk.Label(pc_adjust, text='from').pack(side=tk.LEFT)
            tk.Entry(pc_adjust, textvariable=self.lower_limit,
                     width=5).pack(side=tk.LEFT)
            tk.Label(pc_adjust, text='to').pack(side=tk.LEFT)
            tk.Entry(pc_adjust, textvariable=self.upper_limit,
                     width=5).pack(side=tk.LEFT)

            pc_adjust.pack()
            pc_control.pack()

            # Statistics figures.
            fig_grid = tk.Frame(self)
            self.statsviewer = self.AutoCorr(fig_grid)
            self.statsviewer.grid(row=0, column=0)

            fig_grid.pack()

        def updatemaxval(self):
            # Update the maximum principal component to be
            # included from textbox.
            xlim_min = self.xlim_min
            xlim_max = self.xlim_max
            newvalue = self.upper_limit.get()
            print('Changing upper PC cutoff to: {0}'.format(newvalue))

            if newvalue == '':  # If blank, don't reset value.
                self.upper_limit.set('')
            else:
                try:
                    assert int(
                        newvalue) <= n_components - 1, 'Index exceeds range'
                    assert int(newvalue) > xlim_min[0], 'Max index below min'
                    self.xlim_max[0] = int(newvalue)
                    self.update_figures(xlim_min[0], xlim_max[0])
                except:  # Not a valid number, reset to previous value.
                    print('Not changing upper PC cutoff.')
                    print('upper limit before', self.upper_limit.get())
                    self.upper_limit.set(str(xlim_max[0]))
                    print('upper limit after', self.upper_limit.get())

        def update_min_val(self):
            # Update the minimum component id to be
            # included from textbox.
            xlim_min = self.xlim_min
            xlim_max = self.xlim_max
            newvalue = self.lower_limit.get()
            print('Changing lower PC cutoff to: {0}'.format(newvalue))

            if newvalue == '':  # If blank, don't reset value.
                self.lower_limit.set('x')
            else:
                try:
                    assert int(newvalue) < xlim_max[0], 'Index exceeds maximum'
                    assert int(newvalue) >= 0, 'Min index below 0'
                    self.xlim_min[0] = int(newvalue)
                    self.update_figures(xlim_min[0], xlim_max[0])
                except:  # Not a valid number, reset to previous value.
                    print('Not changing lower PC cutoff.')
                    print('lower limit before', self.lower_limit.get())
                    self.lower_limit.set('x')
                    print('lower limit after', self.lower_limit.get())

        def update_figures(self, xlim_min, xlim_max):
            # Update all figures based on limits set by text update.
            self.statsviewer.update(xlim_min, xlim_max)

            if type(timecourses) is np.ndarray:
                self.component_viewer_frame.update(xlim_min, xlim_max)

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
            # View variance explained by each 'eigenvalue'.
            def __init__(self, parent):
                tk.Frame.__init__(self, parent)

                # Calculate lag autocorrelations.
                x_grid = np.linspace(-0.2, 1.2, 1200)

                frame = tk.Frame(self)
                self.fig = Figure(figsize=(15, 8), dpi=100, frameon=False)
                f = self.fig

                self.ax1 = f.add_subplot(131)
                eig_std = components['timecourses'].std(axis=1)
                self.ax1.plot(eig_std, '.r')
                print('cutoff:', cutoff)
                self.ax1.plot(
                    np.where(lag1[:n_components] > cutoff)[0],
                    eig_std[np.where(lag1[:n_components] > cutoff)[0]], '.b')
                self.ax1.set_xlabel('Independent Component')
                self.ax1.set_ylabel('Timecourse Standard Deviation')

                self.ax2 = f.add_subplot(132)
                self.ax2.set_ylim(-0.2, 1.2)
                f.subplots_adjust(left=0.2, bottom=0.15, right=0.85, top=0.85)
                self.ax2.plot(np.where(lag1 > cutoff)[0],
                              lag1[lag1 > cutoff],
                              'b',
                              label='Lag-1 Signal')
                self.ax2.plot(np.where(lag1 < cutoff)[0],
                              lag1[lag1 < cutoff],
                              'r',
                              label='Lag-1 Noise')
                self.ax2.set_title('Component AutoCorrelation')
                self.ax2.set_ylabel('AutoCorrelation')
                self.ax2.set_xlabel('Independent Component')
                self.ax2.legend()

                self.ax3 = f.add_subplot(133)
                self.ax3.set_ylim(-0.2, 1.2)
                n, _, _ = self.ax3.hist(lag1, bins=50, orientation='horizontal')
                self.ax3.plot(np.exp(log_pdf) * n.max() / np.exp(log_pdf).max(),
                              x_grid,
                              lw=3)
                if cutoff is not None:
                    self.ax3.axhline(cutoff,
                                     color='r',
                                     lw=3,
                                     linestyle='dashed')
                self.ax3.set_title('Lag-1 Histogram')
                self.ax3.set_xlabel('n')

                self.canvas = FigureCanvasTkAgg(f, frame)
                self.canvas.mpl_connect('button_press_event',
                                        lambda event: self.on_key_press(event))
                self.canvas.draw()
                self.canvas.get_tk_widget().pack()
                frame.pack()

            def update(self, xlim_min, xlim_max):
                self.ax1.set_xlim([xlim_min, xlim_max])
                self.ax2.set_xlim([xlim_min, xlim_max])

                self.ax3.cla()

                if 'lag1_full' in components:
                    lag1 = components['lag1_full']
                    _, _, log_pdf = sort_noise(timecourses[xlim_min:xlim_max],
                                               lag1=lag1,
                                               return_logpdf=True)
                else:
                    lag1 = lag_n_autocorr(timecourses, 1, verbose=False)
                    _, _, log_pdf = sort_noise(timecourses[xlim_min:xlim_max],
                                               return_logpdf=True)

                x_grid = np.linspace(-0.2, 1.2, 1200)
                n, _, _ = self.ax3.hist(lag1[xlim_min:xlim_max],
                                        bins=50,
                                        orientation='horizontal')
                self.ax3.plot(np.exp(log_pdf) * n.max() / np.exp(log_pdf).max(),
                              x_grid,
                              lw=3)
                if cutoff is not None:
                    self.ax3.axhline(cutoff,
                                     color='r',
                                     lw=3,
                                     linestyle='dashed')

                self.canvas.draw()

            def on_key_press(self, event):
                if event.button == 3:
                    save_figure(self.fig)

    # Run main loop.
    app = AnalysisGui()
    app.mainloop()
    app.quit()

    # Find which indices to use for reconstruction.
    if toggle[0] == 100:
        raise Exception('Operation was cancelled')

    if toggle.sum() == 0:
        print('All components are classified as signal')
    else:
        print('{0} components classified as signal, {1} '
              'will be used for signal reconstruction.'.format(
                  toggle.sum(), int(toggle.size - toggle.sum())))

    # Update components with toggle info.
    components['artifact_components'] = toggle
    components['cutoff'] = cutoff
    components['noise_components'] = noise_components
    if 'domain_ROIs' in components:
        components['region_assignment'] = region_assignment
        components['region_labels'] = regions

    if load_hdf5:  # If data came from save file, append toggle and timecourses.
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
