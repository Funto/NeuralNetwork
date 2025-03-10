import os
import sys
import subprocess
import xml.etree.ElementTree as ET
import time
from threading import Thread
from queue import Queue
from datetime import timedelta

############ Configuration ############
NB_THREADS=16
OBJ_DIR="emscripten_obj"
DEPENDS_DIR="emscripten_depends"
OUT_DIR="html"
MSVC_PROJ="NeuralNetwork.vcxproj"

COMPILE_ARGS = " -std=c++20"
COMPILE_ARGS += " -Iexternals/imgui-docking"
COMPILE_ARGS += " -Iexternals/stb"
COMPILE_ARGS += " -include src/Globals.h"

#COMPILE_ARGS += " -g -O0 -fdebug-compilation-dir=.."   # DEBUG

LINK_ARGS = " -s USE_WEBGL2=1 -s USE_GLFW=3 -s WASM=1 -s ALLOW_MEMORY_GROWTH"

#LINK_ARGS += " -g -fdebug-compilation-dir=.." # DEBUG: -g is needed here

LINK_ARGS += " --preload-file emscripten_data"

# OTHER ARGUMENTS: -s USE_SDL=2 -s MODULARIZE -s EXPORT_ES6 -sMAX_WEBGL_VERSION=2 -o dist/neuralnetwork.js

############ Utility functions ############
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def get_time_str(seconds):
    td_str = str(timedelta(seconds=seconds))    # "hh:mm:ss"
    x = td_str.split(':')
    if x[0] != "0":
        return x[0] + "h " + x[1] + "m " + x[2] + "s"
    else:
        return x[1] + "m " + x[2] + "s"

def run_cmd(cmd_name, cmd_path, cmd_args):
    print(cmd_name + " " + cmd_args)
    subprocess.call([cmd_path] + cmd_args.split())

def mkdir_safe(path):
    if not os.path.exists(path):
        os.mkdir(path)

############ Setup and print context ############

mkdir_safe(OBJ_DIR)
mkdir_safe(OUT_DIR)
mkdir_safe(DEPENDS_DIR)

if sys.platform != "win32":
    eprint("This script is developed for Windows, running on " + sys.plaform)
    exit(1)

if not 'EMSDK' in os.environ:
    eprint("EMSDK environment variable not found: need to run 'emsdk activate' before executing this script.")
    exit(1)

print("EMSDK=" + os.environ['EMSDK'])
EMCC = os.environ['EMSDK'] + "/upstream/emscripten/emcc.bat"

if not os.path.exists(EMCC):
    eprint("Cannot find emcc (looking for '" + EMCC + "')")
    exit(1)

print("EMCC=" + EMCC)
print("OBJ_DIR=" + OBJ_DIR)
print("OUT_DIR=" + OUT_DIR)

should_rebuild = ('--rebuild' in sys.argv) or ('-r' in sys.argv)
if should_rebuild:
    print("*** Rebuilding all")

should_build_data = ('--data' in sys.argv) or ('-m' in sys.argv)
if should_build_data:
    print("*** Building data")
    os.system("cmd.exe /C build_data_emscripten.bat")

############ Parse Visual Studio project ############
class CppFileInfo:
    def __init__(self, path):
        self.path = path
        self.mtime = os.path.getmtime(path)
        self.hdr_dependencies = []

def parse_vcxproj(path, cpp_fileinfos, hdr_fileinfos):
    xmltree = ET.parse(MSVC_PROJ)
    xmlroot = xmltree.getroot()
    for itemgroup in xmlroot.findall('{http://schemas.microsoft.com/developer/msbuild/2003}ItemGroup'):
        for clcompile in itemgroup.findall('{http://schemas.microsoft.com/developer/msbuild/2003}ClCompile'):
            cpp_fileinfos.append(CppFileInfo(clcompile.attrib['Include']))
    for itemgroup in xmlroot.findall('{http://schemas.microsoft.com/developer/msbuild/2003}ItemGroup'):
        for clinclude in itemgroup.findall('{http://schemas.microsoft.com/developer/msbuild/2003}ClInclude'):
            hdr_fileinfos.append(CppFileInfo(clinclude.attrib['Include']))

print("*** Parsing " + MSVC_PROJ + "...")
cpp_fileinfos = []
hdr_fileinfos = []
parse_vcxproj(MSVC_PROJ, cpp_fileinfos, hdr_fileinfos)
print(str(len(cpp_fileinfos)) + " .cpp files found")
print(str(len(hdr_fileinfos)) + " header files found")

# WIP: need to compare file dates of .dep VS .cpp
#print("*** Collecting header dependencies...")
#for cpp_fileinfo in cpp_fileinfos:
#    cpp_filename = cpp_fileinfo.path.split("\\")[-1]
#    noext_filename = cpp_filename.rsplit(".", 1)[0]
#    depends_filename = noext_filename + ".dep"
#    depends_path = DEPENDS_DIR + "\\" + depends_filename
#
#    cmd_args = cpp_fileinfo.path + " -M -MF " + depends_path + COMPILE_ARGS
#    run_cmd('$(EMCC)', EMCC, cmd_args)

############ Compile ############

print("*** Compiling...")

compile_commands_args_queue = Queue()

nb_cpp_to_compile = 0
nb_cpp_total = len(cpp_fileinfos)

class ObjFileInfo:
    def __init__(self, cpp_fileinfo):
        cpp_filename = cpp_fileinfo.path.split("\\")[-1]
        noext_filename = cpp_filename.rsplit(".", 1)[0]
        obj_filename = noext_filename + ".o"
        obj_path = OBJ_DIR + "\\" + obj_filename

        self.path = obj_path
        self.mtime = 0 if not os.path.exists(obj_path) else os.path.getmtime(obj_path)
        self.need_rebuild=True if cpp_fileinfo.mtime > self.mtime or should_rebuild else False

obj_fileinfos = []
for cpp_fileinfo in cpp_fileinfos:
    obj_fileinfo = ObjFileInfo(cpp_fileinfo)
    obj_fileinfos.append(obj_fileinfo)
    if obj_fileinfo.need_rebuild:
        nb_cpp_to_compile = nb_cpp_to_compile+1
        #print("COMPILE " + cpp_fileinfo.path + " TO " + obj_path)
        cmd_args = cpp_fileinfo.path + " -c -o " + obj_fileinfo.path + COMPILE_ARGS

        # TEMP TEST
        #if cpp_filename != "Game.cpp":
        #if cpp_filename != "Globals.cpp":
        #    continue

        compile_commands_args_queue.put(cmd_args)
        #run_cmd('$(EMCC)', EMCC, cmd_args)

def compilation_consumer_threadfunc(compile_commands_args_queue):
    while True:
        compile_commands_args = compile_commands_args_queue.get()
        run_cmd('$(EMCC)', EMCC, compile_commands_args)
        compile_commands_args_queue.task_done()

time_before_compiling=time.time()
compilation_threads = []
for i in range(NB_THREADS):
    compilation_threads.append(Thread(target=compilation_consumer_threadfunc, args=(compile_commands_args_queue,)))
    compilation_threads[i].daemon = True
    compilation_threads[i].start()

compile_commands_args_queue.join()
    
time_after_compiling=time.time()

############ Link ############

print("*** Linking...")
# TODO: link all object files together
cmd_link_args = ""
for obj_fileinfo in obj_fileinfos:
    cmd_link_args += " " + obj_fileinfo.path
cmd_link_args += ' -o ' + OUT_DIR + '\\neuralnetwork.html' + LINK_ARGS
run_cmd('$(EMCC)', EMCC, cmd_link_args)

time_after_linking=time.time()

print("Compilation took: " + get_time_str(time_after_compiling-time_before_compiling) + " (compiling " + str(nb_cpp_to_compile) + " file(s))")
print("Linking took: " + get_time_str(time_after_linking-time_after_compiling))
print("Total time: " + get_time_str(time_after_linking-time_before_compiling))
