
# Project prt-videoDownloader

Python based server that allows downloading a video using the `yt-dlp` CLI,
and expose the downloaded result via a public HTTPS URL.

[https://github.com/yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp)

A first iteration of the projet requires several calls to the `prt-videoDownloader` API
to initiate the download and get the download URL.

A second iteration of the project should make it possible to initiate the download
and access the downloaded result via a single HTTP GET call (for more convenience when
using it in video workflows).
