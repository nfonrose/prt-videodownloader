#!/bin/bash

# Step000 - Check that expected variables are defined or exit right away with an help message with example values
#
#   export PRT_OPT_BASEPATH=/Users/teevity/Dev/misc/1.prtVideoDownloader/opt/prt/prt-videodownloader
#   export PRT_VIDEODOWNLOAD_PROJECT=/Users/teevity/Dev/misc/1.prtVideoDownloader
#
if [ -z "${PRT_OPT_BASEPATH}" ] || [ -z "${PRT_VIDEODOWNLOAD_PROJECT}" ]; then
  echo -e "\nEnvironment variable not defined for PRT_OPT_BASEPATH=[${PRT_OPT_BASEPATH}] or PRT_VIDEODOWNLOAD_PROJECT=[${PRT_VIDEODOWNLOAD_PROJECT}].\n"
  echo -e "  Example:"
  echo -e "    export PRT_VIDEODOWNLOAD_PROJECT=/home/teevity/Dev/0.perso/prt-videodownloader"
  echo -e "    export PRT_OPT_BASEPATH=/opt/prt/prt-videodownloader\n"
  exit 1
fi

# Step001 - Create folder structure
echo -e "\nStep001 - Creating the  directory structure... ([${PRT_OPT_BASEPATH}/opt/prt])"
