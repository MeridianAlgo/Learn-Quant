# Google News JSON API

### Installation
    npm install --save google-news-json
###### Or
    yarn add google-news-json

### Usage

Usage example:

    let googleNewsAPI = require("google-news-json");
    let news = await googleNewsAPI.getNews(googleNewsAPI.TOP_NEWS, null, "en-GB");

Also supports callback

    googleNewsAPI.getNews(googleNewsAPI.SEARCH, "apple", "en-GB", (err, response) => {
        console.log(response);
    });

### Parameters
Method (defaults to `TOP_NEWS` or `HIGHLIGHTS`)

Query (this is ignored when method is `TOP_NEWS` or `HIGHLIGHTS`)

Locale (defaults to `en-GB`)

Callback (not required)

### Methods
`HIGHLIGHTS`, `TOP_NEWS`, `LOCATION`, `SEARCH`, `TOPIC`, `GEO`

### Supported TOPICS
`TOPICS_WORLD`, `TOPICS_NATION`, `TOPICS_BUSINESS`, `TOPICS_TECHNOLOGY`, `TOPICS_ENTERTAINMENT`, `TOPICS_SCIENCE`, `TOPICS_SPORTS`, `TOPICS_HEALTH`

## Features and bugs

Please file feature requests and bugs at the [issue tracker][tracker].

[tracker]: https://github.com/riftninja/google-news-api/issues

##### If you find this project helpful please<br/>
<a href="https://www.buymeacoffee.com/riftninja" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" ></a>
