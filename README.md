# STB-TESTER V3


**Automated User Interface Testing for Set-Top Boxes & Smart TVs**


## Installation
Cloning a repository :

* Use this command to clone STBTv3 repository:
       
       git clone https://gitlab-avs.rmm.scom/avs/Sagemcom-stbtester-v3.git

* To switch to a tag:
       
       git checkout <tag_name>

Once you're done , you should have a directory containing this tree:

    .
    ├── env_setup.sh
    ├── install.sh
    ├── projects/
    ├── README.md
    ├── run-quad.sh
    ├── stb-tester/
    ├── stbts-run.sh
    ├── thirdparty/
    └── trunk/


* To install required python packages + (GStreamer , IRtrans, Tesseract etc), run this command:


    ./install.sh

**Note:** if this is your first time using stbt on your PC , this step is mandatory.

## Project architecture
Our project architecture is as follows:

![Alt text](https://i.ibb.co/pnPRV7m/archi.png?raw=true "Project architecture")



*  **1- STB-Tester Tag 32**: A package that is provided by the founders of stb-tester
(community). It contains the APIs and the basic functions of STB-Tester. The package should be patched
to support IRTrans, OCR threshold, audio detection etc.

* **2- STB-Tester Sagemcom**: a package developed by Sagemcom, divided into three parts:

* ***Trunk***: Inherits from stb tester (community) and provides more flexible APIs, it is divided into
modules.

* ***Utility***: Contains generic functions not related to stb tester. for example
(Multi-threading, Write/Read CSV etc ...) .

* ***Reporting (stbt-batch)***: Contains modules related to generating and mailing reports.

* ***Projects***: Contains tests-scripts related to a each project.
## How to
stb-tester can be run on two different modes [**Mono/Quad**] :

***Mono :*** 


***I- Running a sanity :***

The simplest way to run a sanity (playlist of tests) is by using this command:

    
    ./stbts-run.sh -p < project name > -c < sanity name > -l < nb_loop > -m < mail >

| Parameter | Description |
|---|---|
    p - project | name of the project (mandatory) 
    c - playlist | playlist name (mandatory)
    l - loop number | specify the number of loops for a sanity (mandatory)
    m - mail | if entered send an mail using the mailing list found in utility mailing_list.txt
    q - quad port | specify quad port (needed for stbt-quad script, you don't have to worry about this parameter) 


* Results will be saved to **/home/STBTSuite/Results**

***II- Running a single test***
To run a single a test, follow these steps:

    1. cd $BO_path
    2. source env_setup.sh <project_name>
    3. stbt run -v test_to_run
    Example : stbt run -v /trunk/generic_tests/youtube/open_youtube_video.py

you can also run a unitary function using this command :

    3. stbt run -v test_file.py::function_name
* Note:
***env_setup.sh*** is a script that will install **[stb tester / trunk]**  to a temporary directory then will load the necessary configuration files according to the selected project. It is also used to set the source_pipeline & sink_pipline according to **[mono/quad]** mode.

***Quad:***

***I - Running sanities on Quad :***

    
    ./run-quad.sh

* Don't forget to fill in project names / sanities / irnetbox / ips etc ... in **run-quad.sh** file first.

***II - Running a single test on quad:***

    1. cd $BO_path
    2. source env_setup.sh -n
    3. stbt quad -p <project 1> -r test_to_run
    
* stbt quad arguments :

    | Parameter | Description |
    |---|---|
     p - projects  | You can pass multiple projects -p project1 -p project2 etc; passing projects only, will open multiple terminals sourced to the specified project. [required]
     c - playlists  |  you can pass multiple playlists each one will be affected to the project (in order)
     t - tv     | run stbt tv on all specified projects.
     s - screenshot | run stbt screenshot on all specified projects.
     r - run        | run a specific test.
     
## Writing test scripts

Testcases are written in the Python programming language. They look like this:

    def test_that_i_can_tune_to_bbc_one_from_the_guide():
        stbt.press("KEY_EPG")
        stbt.wait_for_match("Guide.png")
        stbt.press("KEY_OK")
        stbt.wait_for_match("BBC One.png")
        stbt.wait_for_motion()
        

To get a general understanding on the available stb-tester functions, you should refer to **/trunk/_sc_stbt/** .

* This python package (directory) is divided into 12 modules (for now):

| Module name | Description |
|---|---|
audio | Contains functions related to audio detection
black | Contains functions related to the presence of a black screen in a video frame
color | Contains functions related to color matching
config | Contains functions related to conf files handling
core | Core stbt functionality (press , wait_until, get frame etc ..)
imgutils | Contains functions related to image handling
logging | Debug / warn / test_status functions etc ..
match | Contains functions related to templates matching (OpenCV)
motion | Contains functions related to motion detection
ocr | Contains functions related to OCR & text matching.
navigation | Contains functions related to menus navigation (using template matching + OCR)
transition | Detection & frame-accurate measurement of animations and transitions.

**Recommended workflow for developing tests:** 

1. Create a directory that contain multiple toolboxes (classes), that uses Trunk APIs like sc_stbt.match( ) and
sc_stbt.ocr( ) to extract information from the screen, and it provides a higher-level API specific to the
project.

2. Once the appropriate classes in the toolbox have been defined, testcases can talk about user-facing concepts like
“guide”, “menu”, and “program title”.
All the low-level details like image-matching, OCR, and regions, should be encapsulated inside the toolboxes (classes).

Here's an example of a class called Menu used to navigate in youtube menus :

        class Menu(object):

            class _YoutubeMainMenu(object):
                """
                Define youtube main menu grid
                """
                ...

            def refresh(self):
                return Menu._YoutubeMainMenu()

            @property
            def selection(self):

                menu_region = stbt.Region(x=0, y=25, width=217, height=510)
    
                m = stbt.match(self.cursor,
                               region=menu_region)
    
                for grid in [self.grid, self.grid_up, self.grid_down]:
                    try:
                        text = grid.get(region=m.region).data
                        return sc_stbt.Keyboard.Selection(text, m.region)
                    except IndexError:
                        pass

            def navigate_to(self, target):
                return self._kb.navigate_to(target, page=self)
    
            @staticmethod
            def to_menu():
                """
                a function to open youtube main menu
                """
                for _ in range(5):
                    sc_stbt.press("KEY_LEFT")
                    menu = sc_stbt.wait_until(lambda: sc_stbt.match( base_template + "templates/cursor.png").match)
                    if menu:
                        return menu
                assert False, "Failed to find Youtube Home after pressing KEY_LEFT 5 times"
    
            def back_to_menu(self):
                """
                pressing key back until matching home page
                """
                sc_stbt.press_until_match(key="KEY_BACK",
                                          image=base_template + "templates/cursor.png",
                                          interval_secs=2,
                                          max_presses=5)
    
            def select(self, menu):
        
                """
                Select the specified menu item .
                """
                Menu._YoutubeMainMenu().navigate_to(menu)

   
Now to navigate to the search menu we simply call the class Menu() and the method related to selecting the menu item

        Menu().select("Search")
        
to go back to the main menu we use :

       youtube.Menu().back_to_menu()

**There are many advantages to using classes and unitary functions:**
* Maintenance: Easier and cheaper. If the GUI changes in appearance, you only need to update your
test-pack in one place, not in hundreds of different testcases.

* Clarity: Your testcases are shorter, and the intent of each testcase is clearer.

* Code re-use: Relevant classes for any particular screen can be reused.

## Common issues
Common issues you might encounter :


        1) IRtrans remote control returned unknown error b'**00063 RESULT ERROR: Specified Remote Control not found'
    
   
* **Solution:** REM file  specified in project conf is not loaded, you need to manually copy it into: ***/usr/local/irtrans/remotes***
        
        
        2) error: [Errno 111] Failed to connect to remote control at localhost:21000: [Errno 111] Connection refused

    
* **Solution:** run this command : ***sudo /etc/init.d/irtrans restart*** , or plug & unplug the IRtrans

        3) gst-stream-error-quark: Stream doesn't contain enough data. (4): Stream doesn't contain enough data. gsttypefindelement.c(983): gst_type_find_element_chain_do_typefinding (): /GstPipeline:pipeline1/GstDecodeBin:decodebin0/GstTypeFindElement:typefind: Can't typefind stream\n"b'[2020-07-15 14:04:35.796415 ]
        
* **Solution:** GStreamer is not receiving input data, you need to unplug - plug magewell card , or run these commands:
            
            $ - pkill -f gst-launch-1.0
            $ - pkill -f magewell
            $ - pkill -f stbt
## Creating a new project / Tags: 
**1- To create a new project make sure your directory is structured as below:**


    .
    ├── Makefile
    ├── remotes/
    ├── stbt_"project_name".conf
    ├── project_name/
    ├── _project_name/
    ├── "project_name".sanity
    ├── templates/
    └── tests/

* **Makefile**: 
Makefile used to copy tooboxes to pythonpath.

* **_project_name/** :
This folder contains Toolbox (classes) that include unitary functions used in tests. 

* **remotes/** :
A directory that contains the remotes associated to the project. 
If the rem file gets updated, it will be automatically copied to the irtrans database
**usr/local/irtrans/remotes**

* **project_name/** :
This folder contains the **\__init\__ .py**

* **tests/** :
This directory contains tests calling the unitary functions from toolboxes.

* **templates/** :
Directory that contains templates used to match images.

* **"project_name.sanity"**: 
File used to put the list of tests you want to run.

* **stbt_"projectname".conf**:
Conf file used to specify the value of parameters according to the needs in tests. 

**2- To commit your changes, run these commands:**

        git add <files/directories to commit>
        git commit -m "Bug number-xxxxxx"
        git push

**3- To create a tag, run these commands:**


        git tag <tag_name> -m "Bug number-xxxxxx"
        git push --tags
            
## Documentation
See the [Python API documentation] for more details.

To build your own test rig hardware, and for community-supported documentation
and mailing list, see the [wiki], in particular [Getting Started].


[Python API documentation]: http://stb-tester.com/manual/python-api
[wiki]: https://github.com/stb-tester/stb-tester/wiki
[Getting Started]: https://github.com/stb-tester/stb-tester/wiki/Getting-started-with-stb-tester
