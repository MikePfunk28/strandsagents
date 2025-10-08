## Always check for existing files
Only create a file after scanning the entire codebase

## Only after verifying the code doesnt exist elsewhere
Only after verifying it does not already exist, or does not exist in another file by a different name doing the same thing.  Once verified there is no file already made that handles what you need.  Only then can you create a file.

## Every file is added to project file list, FileList.md.  That should be in the .clinerules folder
If you create a file, add it to the FilesList.md file with a link to the file, and a description of what it does.  If you do not create a file, do not add anything to the FilesList.md file, unless there is no FilesList.md file in the .clinerules directory.

## If there is no FileList.md, create it, initialize the FileList.md
If there is not one, create it, and if a file you discover is not in the list, add it with a link and description.  When you create the FilesList.md file, scan every folder and file from the root on, and add every file in the codebase to the FilesList.md file with a link and description of what it does.

## How it works
You should only have to scan the codebase once immediately and check for a FileList.md, if none initialize it, else check all the files are listed correctly.
Make sure to list every file once, and continually update it whenever there is any change to the codebase, moving, or removing, or creating files.  This way, you can always find any file in the codebase, and know what it does, and where it is located.


## ALWAYS CHECK WORKFLOWS MAIN Workflow.md

Always: "Check the file used, and fix the file used, building upon iterations, and make sure when you do, you document every single change."