package memory

import cai "github.com/anthropic/cai"

func init() {
	cai.DBFactory = func(path string) (cai.MemoryDB, error) {
		return Open(path)
	}
	cai.SystemProber = func(db cai.MemoryDB) error {
		if sdb, ok := db.(*SQLiteDB); ok {
			return ProbeSystem(sdb)
		}
		return nil
	}
}
