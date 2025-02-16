# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-02-16

### Added
- Encryption for AI response data using Fernet encryption
- Added Proejct Scoring form and additional data fields to project table
- Added preview uploaded document and search within document features

### Changed
- Updated response text handling to use encryption
- updated project list view with improved styling of buttons and assessment/review status

### Security
- Added encryption for sensitive response data
- Implemented secure text storage using Fernet encryption
- Maintained consistent encryption approach across document and response data


## [1.1.0] - 2025-02-12
### Added
- Added project_id to evaluation_responses table for better query performance
- Updated response review logic to only count most recent responses
- Response Review toggle and edit functionality

### Changed
- Improved response editing UI with larger, resizable text area
- Version number updated to 1.2.0

### Fixed
- Issue with duplicate question responses appearing in project details
- Response review counting including outdated responses

## [1.0.5] - 2025-02-05
### Added
- New field 'corruption_quintile' to projects table
- Display of corruption quintile in project details view
- Updated data import script to handle new field
- Migration script for adding corruption_quintile column
### Changed
- Added in-place editing for project details table
- Separated project details and notes update functionality
- Improved project details editing with jQuery-based toggle functionality
### Fixed
- Improved null value handling in project details display

## [1.0.0] - 2025-02-01
### Added
- Initial release
- Project management functionality
- Document upload and management
- AI assessment capabilities using Claude
- User authentication system
- API logging
- Project schema with basic fields including:
  - name_of_round (numeric)
  - wb_income_classification
  - cci (Control of Corruption Index)
  - government_type_eiu
  - government_score_eiu 