# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2024-02-05
### Added
- New field 'corruption_quintile' to projects table
- Display of corruption quintile in project details view
- Updated data import script to handle new field
- Migration script for adding corruption_quintile column

## [1.0.0] - 2024-02-01
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