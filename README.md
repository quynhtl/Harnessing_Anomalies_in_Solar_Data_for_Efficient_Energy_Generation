# Quick use
- Step 1: Run command
```
$ docker compose up --build -d --force-recreate
```
- Step 2: Wait 5 minutes - Until container log show something like this
```
$ docker logs -f dash_container

/usr/src/app/app.py:107: SettingWithCopyWarning:

A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead

See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

Dash is running on http://0.0.0.0:8050/

 * Serving Flask app 'app'
 * Debug mode: on
/usr/src/app/app.py:107: SettingWithCopyWarning:


A value is trying to be set on a copy of a slice from a DataFrame.
Try using .loc[row_indexer,col_indexer] = value instead
```
- Step 3: Go to http://127.0.0.1:8050/