# client-code-example
Part of the code I've written to scrape data from the Energy Star Portfolio Manager API in XML format, convert it to dataclasses and convert dataclasses to Django model objects. Django models are veing automatically generated from the dataclasses, that are being generated from the xsd schemas.


Basically the flow of defining models is: xsd schemas -> python classes(dataclasses) -> Django model

Flow of converting of the actual data is: XML data -> python dataclass representation -> Django model object.

Note: The most difficult part here is that API does not send an ID for certain part of the instances, so I had to deal with not duplicating the data.