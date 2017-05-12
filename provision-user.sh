#!/usr/bin/env bash

setup_weka() {
    NEW_FAVORITES=$(dbus-launch gsettings get com.canonical.Unity.Launcher favorites | sed "s/\('application:\/\/firefox.desktop'\)/\1, 'application:\/\/weka.desktop'/")
    dbus-launch gsettings set com.canonical.Unity.Launcher favorites "$NEW_FAVORITES"
}

setup_weka
