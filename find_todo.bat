@echo off

REM This short script will just find all of the lines
REM in the header and source files that contain "// TODO"

findstr /c:"// TODO" /i /n "source\\*.*"
findstr /c:"// TODO" /i /n "include\\*.*"
