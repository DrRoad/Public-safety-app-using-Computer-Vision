import pexpect

def alert():
	child = pexpect.spawn("ssh pi@192.168.43.172 'python3 ~/alert.py > alert_log 2>&1 &'")
	child.expect(".*password: ")
	child.sendline("raspberry")
	child.interact()
