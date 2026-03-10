Vagrant.configure("2") do |config|
  # Ubuntu 22.04 LTS
  config.vm.box = "ubuntu/jammy64"

  # Tên VM
  config.vm.hostname = "ubuntu-2204"

  # SSH
  config.ssh.username = "vagrant"
  config.ssh.insert_key = true

  # Network (private IP – có thể đổi)
  config.vm.network "private_network", ip: "192.168.56.10"

  # Provider VirtualBox
  config.vm.provider "virtualbox" do |vb|
    vb.name = "ubuntu-22.04-50gb"
    vb.memory = 4096
    vb.cpus = 2

    # Resize disk to 50GB
    vb.customize [
      "modifyhd",
      "--resize", 51200,
      "#{Dir.pwd}/.vagrant/machines/default/virtualbox/disk.vdi"
    ]
  end
  # Create user + set password
  config.vm.provision "shell", inline: <<-SHELL
    USERNAME=admin
    PASSWORD=admin123

    # Create user if not exists
    id -u $USERNAME &>/dev/null || sudo useradd -m -s /bin/bash $USERNAME

    # Set password
    echo "$USERNAME:$PASSWORD" | sudo chpasswd

    # Add sudo quyền
    sudo usermod -aG sudo $USERNAME

    # Enable password login for SSH
    sudo sed -i 's/^PasswordAuthentication no/PasswordAuthentication yes/' /etc/ssh/sshd_config
    sudo systemctl restart ssh
  SHELL
end
