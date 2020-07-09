from logo_detector.views import APP # pylint: disable=maybe-no-member

if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=8000, use_reloader=True, debug=True)

