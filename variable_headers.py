DISALLOW_LIST = [b'SIMPLE  = ', b'auto doc=[struct', b'/* XPM *', b'auto doc\x0d\x0a=\x0d\x0a[struct', b'auto saved_art_cache=[struct', b'id=ImageMagick', b'id=MagickCache',
                b'LBLSIZE=', #image formats that look like text
                b'~^\x0d\x0a#ERROR messages', #source code marked up with compiler errors. better to use original.
                b'From: ', b'\x0d\x0a\xc4 Area: ', b'Date: ', b'Received: ', b'Produced by Qmail', b'Produced By O_QWKer', 'To: ', #mail messages. should at least be preprocessed.
                b'.--------------------------------------------------------------------.', b'Session Start: ', b'\x0d\x0aSession Start: ', #instant messages/IRC. should at least be preprocessed.
                b'************************************************************\x0d\x0aMicrosoft Setup Log File Opened', b'\x0d\x0a Volume in drive', b'***  Installation Started',
                b'\xfe   KAV for ', b'FindVirus version', b'Virus scanning report', b'TechFacts 95 System Watch Report', b'Microsoft Office Find Fast Indexer',
                b'*********************************************************************************\x0d\x0a*\x0d\x0a* Log opened:', b'***********  Start log **************'
                b'**********************************************************************\x0d\x0a                            Scan Results',
                b'                               System Information', b'Norman Sandbox Information',
                b'                              File Fix dBASE Repair', b'Microsoft Anti-Virus.', b'Ghost CRC32 Verification list file', #various logs
                b'///////////////////////////////////////////////////////////////////////////////\x0d\x0a// All Platform Dat Update script.', #mcafee ini file
                b'    Offset  String', b'  Offset  String' #output of strings utility
               ]
