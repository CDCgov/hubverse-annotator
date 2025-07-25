# Hubverse Annotator

> [!IMPORTANT]
> This project is currently IN PROGRESS. As such, there may be parts of this repository that do not make much sense or that are broken!

## Background

The `hubverse-annotator` (hereafter, annotator) is a Python `streamlit` application desired to aid CFA's Short Term Forecasts (STF) team with forecast reviews.

The annotator is designed for:

* Visualization of model forecasts and data across location, reference date, and model.
* Selection and exportation of data points for exclusion or comment.
* Selection and exportation of model forecasts for submission to forecast hubs.

## Usage

The annotator can be run by:

* `git clone https://github.com/CDCgov/hubverse-annotator`
* `cd hubverse-annotator`
* `uv run streamlit run ./hubverse_annotator/app.py`

The annotator is designed to work with [hubverse](https://hubverse.io/) data. Examples of this data are available in the [COVID-19 Forecast Hub](https://github.com/CDCgov/covid19-forecast-hub/).

* Observed Data File: see <https://github.com/CDCgov/covid19-forecast-hub/blob/main/target-data/time-series.parquet>
* COVID-19 Hubverse Forecasts File: see <https://github.com/CDCgov/covid19-forecast-hub/tree/main/model-output>

The following is a showcasing of the annotator:

<details markdown=1>

<summary> Demonstration </summary>


https://github.com/user-attachments/assets/fc8d06c0-fd9d-41f7-8fe8-e8069e15af05

</details>


## CDCgov GitHub Organization Protocols

**General disclaimer** This repository was created for use by CDC programs to collaborate on public health related projects in support of the [CDC mission](https://www.cdc.gov/about/cdc/#cdc_about_cio_mission-our-mission).  GitHub is not hosted by the CDC, but is a third party website used by CDC and its partners to share information and collaborate on software. CDC use of GitHub does not imply an endorsement of any one particular service, product, or enterprise.

### Related documents

* [Open Practices](./cdc_protocols/open_practices.md)
* [Rules of Behavior](./cdc_protocols/rules_of_behavior.md)
* [Thanks and Acknowledgements](./cdc_protocols/thanks.md)
* [Disclaimer](DISCLAIMER.md)
* [Contribution Notice](CONTRIBUTING.md)
* [Code of Conduct](./cdc_protocols/code-of-conduct.md)

### Public Domain Standard Notice

This repository constitutes a work of the United States Government and is not subject to domestic copyright protection under 17 USC § 105. This repository is in the public domain within the United States, and copyright and related rights in the work worldwide are waived through the [CC0 1.0 Universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/). All contributions to this repository will be released under the CC0 dedication. By submitting a pull request you are agreeing to comply with this waiver of copyright interest.

### License Standard Notice

The repository utilizes code licensed under the terms of the Apache Software License and therefore is licensed under ASL v2 or later.

This source code in this repository is free: you can redistribute it and/or modify it under the terms of the Apache Software License version 2, or (at your option) any later version.

This source code in this repository is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the Apache Software License for more details.

You should have received a copy of the Apache Software License along with this program. If not, see http://www.apache.org/licenses/LICENSE-2.0.html

The source code forked from other open source projects will inherit its license.

### Privacy Standard Notice

This repository contains only non-sensitive, publicly available data and information. All material and community participation is covered by the [Disclaimer](DISCLAIMER.md) and [Code of Conduct](code-of-conduct.md). For more information about CDC's privacy policy, please visit [http://www.cdc.gov/other/privacy.html](https://www.cdc.gov/other/privacy.html).

### Contributing Standard Notice

Anyone is encouraged to contribute to the repository by [forking](https://help.github.com/articles/fork-a-repo) and submitting a pull request. (If you are new to GitHub, you might start with a [basic tutorial](https://help.github.com/articles/set-up-git).) By contributing to this project, you grant a world-wide, royalty-free, perpetual, irrevocable, non-exclusive, transferable license to all users under the terms of the [Apache Software License v2](http://www.apache.org/licenses/LICENSE-2.0.html) or later.

All comments, messages, pull requests, and other submissions received through CDC including this GitHub page may be subject to applicable federal law, including but not limited to the Federal Records Act, and may be archived. Learn more at [http://www.cdc.gov/other/privacy.html](http://www.cdc.gov/other/privacy.html).

### Records Management Standard Notice

This repository is not a source of government records, but is a copy to increase collaboration and collaborative potential. All government records will be published through the [CDC web site](http://www.cdc.gov).

### Additional Standard Notices

Please refer to [CDC's Template Repository](https://github.com/CDCgov/template) for more information about [contributing to this repository](https://github.com/CDCgov/template/blob/main/CONTRIBUTING.md), [public domain notices and disclaimers](https://github.com/CDCgov/template/blob/main/DISCLAIMER.md), and [code of conduct](https://github.com/CDCgov/template/blob/main/code-of-conduct.md).
