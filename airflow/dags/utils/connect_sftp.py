import pysftp

class connSftp:
    def __init__(self):
        self.host = '183.105.120.175'
        self.port = 2050
        self.username = 'admin' # 서버 유저명
        self.password = 'ishango!23' # 유저 비밀번호
        self.cnopts = pysftp.CnOpts()

        if self.cnopts.hostkeys.lookup(self.host) == None:
            print("Hostkey for " + self.host + " doesn’t exist")
            self.cnopts.hostkeys = None

    def upload(self, path_server, path_local, date):
        with pysftp.Connection(
            self.host,
            port = self.port,
            username = self.username,
            password = self.password,
            cnopts = self.cnopts) as sftp:

            if self.cnopts.hostkeys != None:
                print("New Host. Caching hostkey for " + self.host)
                self.hostkeys.add(self.host, sftp.remote_server_key.get_name(), sftp.remote_server_key) # 호스트와 호스트키를 추가
                print(pysftp.helpers.known_hosts())

                self.hostkeys.save(pysftp.helpers.known_hosts()) # 새로운 호스트 정보 저장

            with sftp.cd(path_server):
                sftp.makedirs(date)
                print(sftp.listdir(''))
                sftp.put_d(path_local, date)
                print(sftp.listdir(''))
