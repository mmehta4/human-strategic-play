# Deploying the arena

The arena is one small Flask app with a SQLite database. Start on the **free** option below.
Move to a paid host only once there's money and the free one becomes too small.

## Before going public (blocking)

- [ ] Ethics approval in hand. Until then, run only a private beta with friends, and don't use the
      data for research.
- [ ] Privacy notice reviewed
- [ ] Hindi and Spanish texts checked by native speakers
- [ ] Name chosen, `site.name` updated in all three language files

## Free: PythonAnywhere (no card needed)

Free plan (checked October 2026; check pythonanywhere.com for current terms): one web app at
`YOURNAME.pythonanywhere.com`, 512 MB disk, one worker. Two things to know:
- **The web app expires after 1 month** unless you click "Run until 1 month from today" on the Web tab. Set a monthly reminder.
- **No custom domain** on the free plan. The free subdomain is fine for launch.

512 MB is enough for several hundred thousand decisions. One worker handles a steady flow of
players, but may slow down during a big launch spike.

1. Sign up at pythonanywhere.com (Beginner, free). Your username becomes the address.
2. Open a **Bash console** (Consoles tab) and run:
   ```bash
   git clone https://github.com/mmehta4/human-strategic-play.git repo
   python3.12 -m venv ~/venv && ~/venv/bin/pip install flask pillow
   ```
   (If the repository is private, clone it with a GitHub personal access token, or upload a zip
   on the Files tab.)
3. Create the secrets file. Replace the values with long random strings: generate them on your
   Mac with `openssl rand -hex 32` and `openssl rand -hex 16`.
   ```bash
   printf 'ARENA_SECRET=PASTE_LONG_RANDOM_1\nARENA_ADMIN_TOKEN=PASTE_LONG_RANDOM_2\n' > ~/.arena_env
   chmod 600 ~/.arena_env
   ```
   **Never change `ARENA_SECRET` after launch:** it would log out every returning player.
4. **Web tab → Add a new web app → Manual configuration → Python 3.12.**
5. On the Web tab:
   - **Virtualenv:** `/home/YOURNAME/venv`
   - **WSGI configuration file:** open it, delete everything, paste the contents of
     `arena/pythonanywhere_wsgi.py`, and replace `YOUR_USERNAME`
   - **Static files:** URL `/static/` → directory `/home/YOURNAME/repo/arena/static`
   - **Force HTTPS:** on
6. Click **Reload**. Open `https://YOURNAME.pythonanywhere.com`, play the whole flow in each
   language, check `/admin/metrics?token=...`, then delete your test data ("Delete my data").
7. **To update the site later:** in a Bash console run `cd ~/repo && git pull`, then click Reload.

### Backups (free)

The database is the single file `/home/YOURNAME/arena.db`. Once a week, or daily during launch,
download `/admin/metrics` numbers and `/export.csv?token=...` to your computer. Keep the export
**private**: never upload it to GitHub or share it.

## Later, when funded: Render or another paid host

Needed for a custom domain, more capacity or always-on reliability. The repository already has a
`Dockerfile`. On Render: New Web Service → repo → root `stage1` → Docker → add a persistent disk
at `/data` → set `ARENA_SECRET` and `ARENA_ADMIN_TOKEN` → health check `/healthz`. Roughly
$7–25 a month; check current pricing.

## Local development

```bash
cd stage1
ARENA_DB=~/arena-dev.db ARENA_SECRET=dev ARENA_ADMIN_TOKEN=dev ~/.venvs/hsb-stage1/bin/python -m arena.app
```

Keep the local database **outside** iCloud-synced folders (`~/arena-dev.db` above). iCloud
syncing a live SQLite file can corrupt it.
