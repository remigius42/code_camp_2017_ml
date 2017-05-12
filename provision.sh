#!/usr/bin/env bash

fix_lc_all_for_bash_rc() {
    BASH_RC_PATH="$1"
    if ! grep --quiet "LC_ALL" "$BASH_RC_PATH" ; then
	logger --stderr "Setting LC_ALL=C in ${BASH_RC_PATH}"
	echo "export LC_ALL=C" >> "${BASH_RC_PATH}"
    fi
}

setup_pip() {
    apt-get install -y python3-pip
    pip3 install --upgrade pip
}

install_quandl_dependencies() {
    apt-get install -y libffi-dev libssl-dev
}

install_matplotlib_dependencies() {
    apt-get install -y python3-tk
}

install_pips() {
    pip3 install NumPy
    pip3 install SciPy
    pip3 install scikit-learn
    pip3 install pandas
    pip3 install quandl
    pip3 install tensorflow
    pip3 install nltk
    pip3 install matplotlib
}

download_nltk_files() {
    python3 -m nltk.downloader -d /usr/local/share/nltk_data book
}

setup_swiss_german_locale() {
    locale-gen de_CH.UTF-8
}

install_java() {
    apt-get install -y openjdk-8-jdk
}

install_desktop_environment() {
    apt-get install -y ubuntu-desktop
}

setup_user_password() {
    echo "ubuntu:ubuntu" | chpasswd
}

install_weka() {
    WEKA_VERSION=3-8-1
    apt-get install -y unzip

    wget --no-verbose http://prdownloads.sourceforge.net/weka/weka-${WEKA_VERSION}.zip
    unzip weka-${WEKA_VERSION} -d /opt
    echo -e "#!/usr/bin/env sh\njava -jar /opt/weka-${WEKA_VERSION}/weka.jar" > /usr/local/bin/weka
    chmod +x /usr/local/bin/weka
}

setup_weka() {
    DESKTOP_DIR=/usr/local/share/applications
    mkdir -p $DESKTOP_DIR
    DESKTOP_FILE=$DESKTOP_DIR/weka.desktop
    cat > $DESKTOP_FILE <<EOF
[Desktop Entry]
Version=1.0
Name=Weka 3.8.1
Comment=Waikato Environment for Knowledge Analysis (WEKA)
Exec=/usr/local/bin/weka
Terminal=false
Icon=/opt/weka-3-8-1/weka.ico
Type=Application
Categories=Application;
EOF
    chmod +x $DESKTOP_FILE
}

apt-get update

fix_lc_all_for_bash_rc /root/.bashrc
fix_lc_all_for_bash_rc /home/ubuntu/.bashrc

setup_pip
install_quandl_dependencies
install_matplotlib_dependencies
install_pips
download_nltk_files
setup_swiss_german_locale

install_java
install_desktop_environment
setup_user_password
install_weka
setup_weka
