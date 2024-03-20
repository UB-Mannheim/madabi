# MaDaBi - Mannheim Data Bibliography

This repo contains work in progress on creating Mannheim Data Bibliography (or datagraphy), i.e., a registry of metadata of all data, created or collected by the employees of the University of Mannheim.

## Motivation

* FAIRness (Findability, Accessibility, Interoperability, and Reusability) of data at the University of Mannheim,
* a single point of access to metadata of data, created or collected by the employees of the University of Mannheim ([MADATA](https://madata.bib.uni-mannheim.de)),
* performance evaluation using metrics for data sharing, data reuse and data citation in the University of Mannheim,
* improving culture of data sharing, data reuse and data citation in the University of Mannheim.

## Problem statement

Mannheim Data Bibliography [MADATA](https://madata.bib.uni-mannheim.de) contains only a part of all (meta)data published by the employees of the University of Mannheim. 
All other data, created or collected by employees of the University of Mannheim, 
are stored somewhere else.

We want to:
* collect metadata of `data`, created or collected by employees of University of Mannheim,
* update the data bibliography regularly,
* store the collected metadata in MADATA,
* evaluate metrics for data sharing, data reuse and data citation,
* create data dashboard.

## Scope of work: Resource Types

What can `data` mean?
1. dataset,
2. software (code, script, package),
3. executable notebook (Jupyter notebook),
4. data management plan, 
5. software management plan,
6. workflow,
7. model,
8. figure,
9. table,
10. image,
11. video,
12. text,
13. interview,
14. project (e.g., https://doi.org/10.3886/E124902V2),
15. reproducibility (or replication) package.

Controlled vocabulary for resource types of [da|ra](https://www.da-ra.de/media/pages/downloads/metadata/version-4-0/87b5e1dc34-1701081033/dara-resourceType-v4.1.xsd)
* Audiovisual
* Collection
* DataPaper
* Dataset
* Event
* Image
* InteractiveResource
* Model
* PhysicalObject
* Service
* Software
* Sound
* Text
* Workflow
* Other

The [resource types for DataCite](https://support.datacite.org/docs/what-are-the-resource-types-for-datacite-dois) DOIs:
* Audiovisual
* Book
* BookChapter
* Collection
* ComputationalNotebook
* ConferencePaper
* ConferenceProceeding
* DataPaper
* Dataset
* Dissertation
* Event
* Image
* InteractiveResource
* Journal
* JournalArticle
* Model
* OutputManagementPlan
* PeerReview
* PhysicalObject
* Preprint
* Report
* Service
* Software
* Sound
* Standard
* Text
* Workflow
* Other

## Scope of work: Madoc and Uni-resources

Scope of work and plan:
* Extract dataset mentions, "Data availability statements" and "Supplemented Materials" from publications at [MADOC](https://madoc.bib.uni-mannheim.de), i.e., Mannheim University bibliography and publication server:
   * From full texts in PDF-files at MADOC
   * From external online versions of publications (using DOIs from MADOC)
* Add metadata for data resources (e.g., databases, digital editions, etc.) hosted at uni-mannheim.de to MADATA
   * [Digital Editions in MaObjects](https://www.bib.uni-mannheim.de/en/lehren-und-forschen/fdz-home/fdz-services/maobjects-1)

## Scope of work: Repositories and portals

Plan:

* Harvesting metadata of data from data repositories and metadata portals
   * Searching for "University of Mannheim" or "Mannheim University" or "Universität Mannheim" :hourglass_flowing_sand:
   * Searching for names of employees of University of Mannheim :hourglass_flowing_sand:

Repositories and portals:
1. External data repositories with links to search queries "University of Mannheim":
   * [Zenodo](https://zenodo.org/search?q=type%3A%28dataset%20OR%20software%20OR%20image%20OR%20video%20OR%20physicalobject%20OR%20datamanagementplan%20OR%20softwaremanagementplan%20OR%20softwaredocumentation%29%20%20AND%20creators.affiliation%3A%28%22University%20of%20Mannheim%22%20OR%20%22Mannheim%20University%22%20OR%20%22Universit%C3%A4t%20Mannheim%22%29&l=list&p=1&s=10&sort=bestmatch)
   * [GESIS Search](https://search.gesis.org/?source=%7B%22query%22%3A%7B%22bool%22%3A%7B%22must%22%3A%5B%7B%22query_string%22%3A%7B%22query%22%3A%22%5C%22Universit%C3%A4t%20Mannheim%5C%22%22%2C%22default_operator%22%3A%22AND%22%7D%7D%5D%2C%22filter%22%3A%5B%7B%22term%22%3A%7B%22type%22%3A%22research_data%22%7D%7D%5D%7D%7D%7D)
   * [Dataverse](https://dataverse.harvard.edu/dataverse/harvard?q=%22University+of+Mannheim%22&types=dataverses&sort=score&order=desc&page=1)
   * [Figshare](https://figshare.com/search?q=%22Mannheim%20Universit%C3%A4t%22&itemTypes=3,1,6)
   * [GitHub](https://github.com/search?q=%22University+of+Mannheim%22&type=repositories) and GitLab
      * https://github.com/dwslab
      * https://github.com/wbsg-uni-mannheim
      * https://github.com/socialsciencedatalab
      * https://github.com/UB-Mannheim
      * https://gitlab.uni-mannheim.de/processanalytics
   * [CESSDA Data Catalogue](https://datacatalogue.cessda.eu/?q=%22University%20of%20Mannheim%22)
   * repositories described in subject-specific publication policies

2. Metadata portals with links to search queries "University of Mannheim":
   * [BASE](https://api.base-search.net/cgi-bin/BaseHttpSearchInterface.fcgi?func=PerformSearch&query=(University%20of%20Mannheim%20AND%20dctypenorm:7))
   * [Google Dataset Search](https://datasetsearch.research.google.com/search?src=0&query=%22university%20of%20mannheim%22)
   * [OpenAIRE Search](https://explore.openaire.eu/search/find?type=%22datasets%22,%22software%22&resultbestaccessright=%22Open%2520Access%22&fv0=%22university%20of%20mannheim%22&f0=q)

## Harvested Metadata

See folder `./metadata/`.
