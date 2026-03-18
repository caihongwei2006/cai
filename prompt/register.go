package prompt

import cai "github.com/anthropic/cai"

func init() {
	cai.PromptLoader = func(dir string, db cai.MemoryDB) {
		prompts, err := LoadAll(dir)
		if err != nil || len(prompts) == 0 {
			return
		}
		SeedToDB(prompts, db)
	}

	cai.PromptVersionWriter = func(dir string, action string, mem cai.IntentMemory) {
		WriteVersion(dir, action, mem)
	}

	cai.WorkspaceDocLoader = func(dir string, ds cai.WorkspaceDocStore) {
		docs, err := LoadWorkspaceDocs(dir)
		if err != nil || len(docs) == 0 {
			return
		}
		SeedDocsToDB(docs, ds)
	}

	cai.WorkspaceDocWriter = func(dir string, name string, doc cai.WorkspaceDocument) {
		WriteDocumentVersion(dir, name, doc)
	}
}
