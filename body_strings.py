DISALLOW_LIST = ['"$CHICAGO$"', '"$Windows NT$"', ' Msg#: ', '$$   Processing started on ', '\x00\x00']
#                 Windows inf files,               mail,      logs (Integrity Master),       generic binaries
DISALLOW_LIST = [s.lower() for s in DISALLOW_LIST]
