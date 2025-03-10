rem Copy files

set inDir=%~dp0\data
set outDir=%~dp0\emscripten_data

rmdir /S /Q "%outDir%"
if not exist "%outDir%" mkdir "%outDir%"

robocopy "%inDir%" %outDir% *-ubyte %roboCopyOpt% /S
robocopy "%inDir%" %outDir% *.bin %roboCopyOpt% /S
