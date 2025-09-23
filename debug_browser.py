from strands_tools.browser import LocalChromiumBrowser
import inspect

# Debug what's available
browser = LocalChromiumBrowser()
print("Browser object:", browser)
print("Browser attributes:", dir(browser))
print("Browser type:", type(browser))

if hasattr(browser, 'browser'):
    print("browser.browser:", browser.browser)
    print("browser.browser type:", type(browser.browser))